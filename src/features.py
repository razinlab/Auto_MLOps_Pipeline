import pandas as pd
import numpy as np
import ta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_price_features(df):
    df_features = df.copy()

    df_features['daily_return'] = df_features['close'].pct_change()

    df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))

    df_features['volatility_5d'] = df_features['daily_return'].rolling(window=5).std()
    df_features['volatility_20d'] = df_features['daily_return'].rolling(window=20).std()

    df_features['momentum_5d'] = df_features['daily_return'].rolling(window=5).sum()
    df_features['momentum_20d'] = df_features['daily_return'].rolling(window=20).sum()

    # Calculate high-low range
    df_features['hl_range'] = (df_features['high'] - df_features['low']) / df_features['close']
    
    return df_features

def add_moving_averages(df):
    df_features = df.copy()

    # Calculate simple moving averages
    df_features['sma_5'] = df_features['close'].rolling(window=5).mean()
    df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
    df_features['sma_50'] = df_features['close'].rolling(window=50).mean()
    df_features['sma_200'] = df_features['close'].rolling(window=200).mean()

    # Calculate exponential moving averages
    df_features['ema_5'] = df_features['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_20'] = df_features['close'].ewm(span=20, adjust=False).mean()
    df_features['ema_50'] = df_features['close'].ewm(span=50, adjust=False).mean()
    df_features['ema_200'] = df_features['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate moving average crossovers
    df_features['sma_5_20_cross'] = (df_features['sma_5'] > df_features['sma_20']).astype(int)
    df_features['sma_20_50_cross'] = (df_features['sma_20'] > df_features['sma_50']).astype(int)
    df_features['sma_50_200_cross'] = (df_features['sma_50'] > df_features['sma_200']).astype(int)
    
    # Calculate price relative to moving averages
    df_features['price_sma_5_ratio'] = df_features['close'] / df_features['sma_5']
    df_features['price_sma_20_ratio'] = df_features['close'] / df_features['sma_20']
    df_features['price_sma_50_ratio'] = df_features['close'] / df_features['sma_50']
    df_features['price_sma_200_ratio'] = df_features['close'] / df_features['sma_200']
    
    return df_features

def add_technical_indicators(df):
    df_features = df.copy()
    
    # RSI (Relative Strength Index)
    df_features['rsi_14'] = ta.momentum.RSIIndicator(df_features['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df_features['close'])
    df_features['macd'] = macd.macd()
    df_features['macd_signal'] = macd.macd_signal()
    df_features['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df_features['close'])
    df_features['bollinger_high'] = bollinger.bollinger_hband()
    df_features['bollinger_low'] = bollinger.bollinger_lband()
    df_features['bollinger_width'] = (df_features['bollinger_high'] - df_features['bollinger_low']) / df_features['close']
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close'])
    df_features['stoch_k'] = stoch.stoch()
    df_features['stoch_d'] = stoch.stoch_signal()
    
    # Average Directional Index (ADX)
    adx = ta.trend.ADXIndicator(df_features['high'], df_features['low'], df_features['close'])
    df_features['adx'] = adx.adx()
    
    # On-Balance Volume (OBV)
    df_features['obv'] = ta.volume.OnBalanceVolumeIndicator(df_features['close'], df_features['volume']).on_balance_volume()
    
    return df_features

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    df_features = df.copy()

    features_to_lag = ['close', 'daily_return', 'volume', 'rsi_14', 'macd', 'adx']

    for feature in features_to_lag:
        if feature in df_features.columns:
            for lag in lags:
                df_features[f'{feature}_lag_{lag}'] = df_features[feature].shift(lag)
    
    return df_features

def add_target(df, horizon=1):
    df_features = df.copy()

    df_features[f'future_return_{horizon}d'] = df_features['close'].pct_change(horizon).shift(-horizon)

    df_features[f'target_{horizon}d'] = (df_features[f'future_return_{horizon}d'] > 0).astype(int)
    
    return df_features

def prepare_features(df, prediction_horizon=1):

    logger.info("Preparing features...")

    df_features = add_price_features(df)
    df_features = add_moving_averages(df_features)
    df_features = add_technical_indicators(df_features)
    df_features = add_lagged_features(df_features)
    df_features = add_target(df_features, horizon=prediction_horizon)

    df_features = df_features.dropna()

    object_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        logger.warning(f"Dropping non-numeric columns: {object_cols}")
        df_features = df_features.drop(columns=object_cols)

    logger.info(f"Feature preparation complete. {len(df_features)} rows remaining after dropping NaN values.")

    return df_features

if __name__ == "__main__":
    from data_ingestion import load_from_parquet
    
    try:
        ticker = 'SPY'
        df = load_from_parquet(ticker)

        df_features = prepare_features(df)

        print(f"Original data shape: {df.shape}")
        print(f"Feature data shape: {df_features.shape}")
        print(f"Features created: {len(df_features.columns) - len(df.columns)}")
        print("\nSample of features:")
        print(df_features.iloc[-1][['close', 'daily_return', 'rsi_14', 'macd', 'target_1d']].to_dict())
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")