"""
Trading module for executing trades based on model predictions.
This module loads models from the MLflow Model Registry, makes predictions,
and submits orders to Alpaca based on those predictions.
"""

import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
import mlflow
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from auth import get_trading_client, get_data_client
from data_ingestion import fetch_historical_data
from features import prepare_features
from model_registry import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'trade_logs.db'

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create trades table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticker TEXT,
        model_type TEXT,
        model_version TEXT,
        prediction INTEGER,
        order_id TEXT,
        order_side TEXT,
        quantity REAL,
        price REAL
    )
    ''')
    
    # Create performance table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        model_type TEXT,
        model_version TEXT,
        ticker TEXT,
        pnl REAL,
        return REAL,
        cumulative_return REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info("Database setup complete")

def log_trade(ticker, model_type, model_version, prediction, order_id, order_side, quantity, price):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO trades (timestamp, ticker, model_type, model_version, prediction, order_id, order_side, quantity, price)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, ticker, model_type, model_version, prediction, order_id, order_side, quantity, price))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Logged {order_side} trade for {ticker} using {model_type} model")

def log_performance(date, model_type, model_version, ticker, pnl, ret, cumulative_return):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO performance (date, model_type, model_version, ticker, pnl, return, cumulative_return)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (date, model_type, model_version, ticker, pnl, ret, cumulative_return))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Logged performance for {ticker} using {model_type} model: PnL=${pnl:.2f}, Return={ret:.2%}")

def get_current_data(tickers):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    data = fetch_historical_data(tickers, start_date_str, end_date_str)

    for ticker in tickers:
        if ticker in data:
            data[ticker] = prepare_features(data[ticker])
    
    return data


def load_models(model_name="daily-price-forecaster"):
    try:
        champion_model = load_model(model_name, alias="champion")
        logger.info(f"Loaded champion model with 'champion' alias")

        challenger_model = load_model(model_name, alias="challenger")
        logger.info(f"Loaded challenger model with 'challenger' alias")

        return champion_model, challenger_model

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def get_model_assignment(tickers):
    mid = len(tickers) // 2
    champion_tickers = tickers[:mid]
    challenger_tickers = tickers[mid:]
    
    logger.info(f"Champion model assigned to: {champion_tickers}")
    logger.info(f"Challenger model assigned to: {challenger_tickers}")
    
    return champion_tickers, challenger_tickers


def make_prediction(model, data, features=None):
    if features is None:
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                        'future_return_1d', 'target_1d']

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col not in exclude_cols]

    latest_data = data.iloc[-1:][features]
    prediction = model.predict(latest_data)[0]
    return int(prediction)


def submit_order(ticker, side, quantity, model_type, model_version, prediction):
    client = get_trading_client()
    order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

    order_request = MarketOrderRequest(
        symbol=ticker,
        qty=quantity,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )

    try:
        order = client.submit_order(order_request)
        logger.info(f"Submitted {side} order for {quantity} shares of {ticker}")

        log_trade(
            ticker=ticker,
            model_type=model_type,
            model_version=model_version,
            prediction=prediction,
            order_id=str(order.id),  # ← This fix
            order_side=side,
            quantity=float(quantity),
            price=0.0
        )

        return order

    except Exception as e:
        logger.error(f"Error submitting order: {e}")
        raise


def run_trading_strategy(tickers, model_name="daily-price-forecaster", cash_per_position=10000):
    setup_database()

    # Load models
    champion_model, challenger_model = load_models(model_name)
    champion_tickers, challenger_tickers = get_model_assignment(tickers)

    start_date = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"Fetching historical data for {tickers} from {start_date} to {end_date}")
    data = fetch_historical_data(tickers, start_date, end_date)

    for ticker in tickers:
        if ticker in data:
            data[ticker] = prepare_features(data[ticker])

    client = get_trading_client()
    account = client.get_account()
    logger.info(f"Account: ${float(account.cash):.2f} cash available")

    for ticker in champion_tickers:
        if ticker not in data:
            logger.warning(f"No data available for {ticker}. Skipping.")
            continue

        if len(data[ticker]) == 0:
            logger.warning(f"No valid data for {ticker} after feature preparation. Skipping.")
            continue

        prediction = make_prediction(champion_model, data[ticker])

        try:
            position = client.get_open_position(ticker)
            current_position = float(position.qty)
        except:
            current_position = 0

        latest_price = data[ticker]['close'].iloc[-1]
        target_position = cash_per_position / latest_price if prediction == 1 else 0
        order_quantity = abs(target_position - current_position)

        if order_quantity > 0:
            if target_position > current_position:
                submit_order(
                    ticker=ticker,
                    side='buy',
                    quantity=order_quantity,
                    model_type='Champion',
                    model_version='Production',
                    prediction=prediction
                )
            else:
                submit_order(
                    ticker=ticker,
                    side='sell',
                    quantity=order_quantity,
                    model_type='Champion',
                    model_version='Production',
                    prediction=prediction
                )

    for ticker in challenger_tickers:
        if ticker not in data:
            logger.warning(f"No data available for {ticker}. Skipping.")
            continue

        if len(data[ticker]) == 0:
            logger.warning(f"No valid data for {ticker} after feature preparation. Skipping.")
            continue

        prediction = make_prediction(challenger_model, data[ticker])

        try:
            position = client.get_open_position(ticker)
            current_position = float(position.qty)
        except:
            current_position = 0

        latest_price = data[ticker]['close'].iloc[-1]
        target_position = cash_per_position / latest_price if prediction == 1 else 0
        order_quantity = abs(target_position - current_position)

        if order_quantity > 0:
            if target_position > current_position:
                submit_order(
                    ticker=ticker,
                    side='buy',
                    quantity=order_quantity,
                    model_type='Challenger',
                    model_version='Staging',
                    prediction=prediction
                )
            else:
                submit_order(
                    ticker=ticker,
                    side='sell',
                    quantity=order_quantity,
                    model_type='Challenger',
                    model_version='Staging',
                    prediction=prediction
                )


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")

    tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'META', 'NVDA']
    
    try:
        run_trading_strategy(tickers)
        
    except Exception as e:
        logger.error(f"Error in trading strategy: {e}")