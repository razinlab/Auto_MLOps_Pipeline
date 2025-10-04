import os
import time
import logging
import argparse
import mlflow
from datetime import datetime, timedelta

from data_ingestion import fetch_historical_data, save_to_parquet
from backtest import BacktestStrategy
from model_registry import register_model, promote_model_to_production, setup_mlflow
from monitor import create_baseline
from trade import run_trading_strategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_data(tickers):
    logger.info("Step 1: Fetching historical data...")

    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = fetch_historical_data(tickers, start_date, end_date)
        save_to_parquet(data)
        logger.info("Data fetching completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return False

def train_and_backtest(ticker):
    logger.info(f"Step 2: Training and backtesting model for {ticker}...")
    
    try:
        backtester = BacktestStrategy(
            ticker=ticker,
            train_size=252,  # 1 year of training data
            prediction_horizon=1  # Predict next day's movement
        )
        
        metrics = backtester.run_backtest(log_mlflow=True)

        client = setup_mlflow()
        runs = mlflow.search_runs(experiment_names=["trading-backtest"])
        if len(runs) == 0:
            logger.error("No runs found in MLflow.")
            return None
        
        latest_run = runs.iloc[0]
        run_id = latest_run.run_id
        
        logger.info(f"Model training and backtesting completed successfully. Run ID: {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"Error training and backtesting model: {e}")
        return None


def register_and_promote_model(run_id, model_name):
    logger.info(f"Step 3: Registering and promoting model {model_name}...")
    try:
        model_version = register_model(run_id, model_name)

        promote_model_to_production(
            model_name=model_name,
            version=model_version.version,
            archive_existing=True
        )

        logger.info(f"Model {model_name} version {model_version.version} promoted successfully")
        return model_version.version

    except Exception as e:
        logger.error(f"Error registering and promoting model: {e}")
        raise


def create_drift_baseline(tickers):
    logger.info("Step 4: Creating baseline for drift detection...")
    
    try:
        for ticker in tickers:
            create_baseline([ticker])
        
        logger.info("Drift baseline creation completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating drift baseline: {e}")
        return False

def start_trading(tickers, model_name):
    logger.info("Step 5: Starting trading...")
    
    try:
        run_trading_strategy(tickers, model_name)
        return True
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        return False

def run_pipeline(tickers, model_name="daily-price-forecaster", wait_time=60):
    logger.info(f"Starting MLOps pipeline for tickers: {tickers}")

    if not fetch_data(tickers):
        logger.error("Pipeline failed at data fetching step.")
        return False

    run_id = train_and_backtest(tickers[0])
    if not run_id:
        logger.error("Pipeline failed at model training and backtesting step.")
        return False

    if not register_and_promote_model(run_id, model_name):
        logger.error("Pipeline failed at model registration and promotion step.")
        return False

    if not create_drift_baseline(tickers):
        logger.error("Pipeline failed at drift baseline creation step.")
        return False

    logger.info(f"Waiting for {wait_time} seconds before starting trading...")
    time.sleep(wait_time)

    if not start_trading(tickers, model_name):
        logger.error("Pipeline failed at trading step.")
        return False
    
    logger.info("MLOps pipeline completed successfully.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the MLOps pipeline')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'AAPL', 'TSLA', 'MSFT', 'AMZN'],
                        help='List of tickers to process')
    parser.add_argument('--model-name', default='daily-price-forecaster',
                        help='Name of the model to register')
    parser.add_argument('--wait-time', type=int, default=60,
                        help='Time to wait (in seconds) before starting trading')
    
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        run_pipeline(args.tickers, args.model_name, args.wait_time)
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")