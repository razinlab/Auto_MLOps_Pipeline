import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import logging
import json

from data_ingestion import load_from_parquet
from features import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestStrategy:
    def __init__(self, ticker, start_date=None, end_date=None, train_size=252, prediction_horizon=1):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_size = train_size
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.data = None
        self.predictions = None
        self.portfolio = None
        self.metrics = {}

    def load_data(self):
        logger.info(f"Loading data for {self.ticker}...")

        df = load_from_parquet(self.ticker)

        if self.start_date:
            df = df[df['date'] >= pd.to_datetime(self.start_date)]
        if self.end_date:
            df = df[df['date'] <= pd.to_datetime(self.end_date)]

        self.data = prepare_features(df, prediction_horizon=self.prediction_horizon)

        logger.info(f"Data loaded and prepared. Shape: {self.data.shape}")

        return self.data

    def train_model(self, train_data, features=None, target=None):
        if features is None:
            # Exclude non-numeric and target columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                            f'future_return_{self.prediction_horizon}d',
                            f'target_{self.prediction_horizon}d']

            # Get only numeric columns
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in exclude_cols]

        if target is None:
            target = f'target_{self.prediction_horizon}d'

        X = train_data[features]
        y = train_data[target]

        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError(f"Non-numeric columns detected: {X.select_dtypes(exclude=[np.number]).columns.tolist()}")

        model = XGBClassifier(
            n_estimators=30,
            max_depth=1,
            learning_rate=0.1,

            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,

            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,

            scale_pos_weight=1.0,

            tree_method='hist',
            random_state=42,
            n_jobs=-1,

            eval_metric='logloss',
        )

        model.fit(X, y)

        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return model

    def backtest_rolling(self, features=None, target=None):
        if self.data is None:
            self.load_data()

        if features is None:
            # Match the same logic as train_model()
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                            f'future_return_{self.prediction_horizon}d',
                            f'target_{self.prediction_horizon}d']

            # Get only numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in exclude_cols]

        if target is None:
            target = f'target_{self.prediction_horizon}d'

        self.predictions = self.data.copy()
        self.predictions['prediction'] = np.nan
        self.predictions['position'] = 0
        self.predictions['portfolio_value'] = 100.0

        min_train_idx = self.train_size

        for i in range(min_train_idx, len(self.data)):
            train_data = self.data.iloc[i - self.train_size:i]
            model = self.train_model(train_data, features, target)

            X_test = self.data.iloc[i:i + 1][features]
            prediction = model.predict(X_test)[0]

            self.predictions.iloc[i, self.predictions.columns.get_loc('prediction')] = prediction
            self.predictions.iloc[i, self.predictions.columns.get_loc('position')] = prediction

        self.predictions['return'] = self.predictions['daily_return'] * self.predictions['position'].shift(1)
        self.predictions['portfolio_return'] = self.predictions['return'].fillna(0) + 1
        self.predictions['portfolio_value'] = 100 * self.predictions['portfolio_return'].cumprod()

        self.model = model
        return self.predictions

    def calculate_metrics(self):
        if self.predictions is None:
            logger.error("No predictions available. Run backtest_rolling() first.")
            return {}

        pred_data = self.predictions.dropna(subset=['prediction'])

        y_true = pred_data[f'target_{self.prediction_horizon}d']
        y_pred = pred_data['prediction']

        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1_score'] = f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()

        portfolio_returns = self.predictions['return'].dropna()

        self.metrics['cumulative_return'] = self.predictions['portfolio_value'].iloc[-1] / self.predictions['portfolio_value'].iloc[0] - 1

        n_years = len(portfolio_returns) / 252
        self.metrics['annualized_return'] = (1 + self.metrics['cumulative_return']) ** (1 / n_years) - 1

        self.metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)

        self.metrics['sharpe_ratio'] = self.metrics['annualized_return'] / self.metrics['volatility'] if self.metrics['volatility'] > 0 else 0

        cum_returns = (1 + portfolio_returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns / peak - 1)
        self.metrics['max_drawdown'] = drawdown.min()

        self.metrics['win_rate'] = (portfolio_returns > 0).mean()

        self.metrics['n_trades'] = (self.predictions['position'] != self.predictions['position'].shift(1)).sum()

        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                self.metrics[key] = float(value)

        return self.metrics

    def plot_results(self, save_path=None):
        if self.predictions is None:
            logger.error("No predictions available. Run backtest_rolling() first.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Price and positions
        ax1.plot(self.predictions['date'], self.predictions['close'], label='Price')
        buy_signals = self.predictions[self.predictions['position'] == 1]
        sell_signals = self.predictions[self.predictions['position'] == 0]
        ax1.scatter(buy_signals['date'], buy_signals['close'], color='green', marker='^', alpha=0.7, label='Buy')
        ax1.scatter(sell_signals['date'], sell_signals['close'], color='red', marker='v', alpha=0.7, label='Sell')
        ax1.set_title(f'{self.ticker} Price and Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        # Portfolio value
        ax2.plot(self.predictions['date'], self.predictions['portfolio_value'], label='Strategy')
        buy_hold = self.predictions['close'] / self.predictions['close'].iloc[self.train_size] * 100
        ax2.plot(self.predictions['date'], buy_hold, label='Buy & Hold', alpha=0.7)
        ax2.set_title('Portfolio Value')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True)

        # Feature importance
        if hasattr(self, 'feature_importance'):
            top_features = self.feature_importance.head(10)
            ax3.barh(top_features['feature'], top_features['importance'])
            ax3.set_title('Top 10 Feature Importance')
            ax3.set_xlabel('Importance')
            ax3.invert_yaxis()
            ax3.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def save_model(self, path='models'):
        if self.model is None:
            logger.error("No model available. Run backtest_rolling() first.")
            return None

        os.makedirs(path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(path, f"{self.ticker}_model_{timestamp}.pkl")

        mlflow.sklearn.save_model(self.model, model_path)

        if hasattr(self, 'feature_importance'):
            feature_path = os.path.join(path, f"{self.ticker}_features_{timestamp}.parquet")
            self.feature_importance.to_parquet(feature_path, index=False)

        if self.metrics:
            metrics_path = os.path.join(path, f"{self.ticker}_metrics_{timestamp}.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)

        logger.info(f"Model saved to {model_path}")

        return model_path

    def run_backtest(self, log_mlflow=True):
        self.load_data()

        self.backtest_rolling()

        metrics = self.calculate_metrics()

        if log_mlflow:
            experiment_name = "trading-backtest"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{self.ticker}_backtest"):
                mlflow.log_param("ticker", self.ticker)
                mlflow.log_param("train_size", self.train_size)
                mlflow.log_param("prediction_horizon", self.prediction_horizon)

                for key, value in metrics.items():
                    if key != 'confusion_matrix':  # Skip non-scalar values
                        mlflow.log_metric(key, value)

                mlflow.sklearn.log_model(self.model, "model")

                if hasattr(self, 'feature_importance'):
                    mlflow.log_dict(self.feature_importance.to_dict(), "feature_importance.json")

                if 'confusion_matrix' in metrics:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = np.array(metrics['confusion_matrix'])
                    ax.imshow(cm, cmap='Blues')
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, cm[i, j], ha='center', va='center')
                    plt.tight_layout()
                    mlflow.log_figure(fig, "confusion_matrix.png")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.predictions['date'], self.predictions['portfolio_value'], label='Strategy')
                buy_hold = self.predictions['close'] / self.predictions['close'].iloc[self.train_size] * 100
                ax.plot(self.predictions['date'], buy_hold, label='Buy & Hold', alpha=0.7)
                ax.set_title('Portfolio Value')
                ax.set_ylabel('Value ($)')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                mlflow.log_figure(fig, "portfolio_value.png")

        print("\n" + "="*50)
        print(f"Backtest Results for {self.ticker}")
        print("="*50)
        print(f"Period: {self.predictions['date'].min().date()} to {self.predictions['date'].max().date()}")
        print(f"Number of trading days: {len(self.predictions.dropna(subset=['prediction']))}")
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nFinancial Performance Metrics:")
        print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Volatility (Annualized): {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Number of Trades: {metrics['n_trades']}")
        print("="*50)

        return metrics

if __name__ == "__main__":

    mlflow.set_tracking_uri("file:./mlruns")

    ticker = 'SPY'

    try:
        backtester = BacktestStrategy(
            ticker=ticker,
            train_size=252,  # 1 year of training data
            prediction_horizon=1  # Predict next day's movement
        )

        backtester.run_backtest()
        backtester.plot_results(save_path=f"models/{ticker}_backtest_results.png")
        backtester.save_model()

    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
