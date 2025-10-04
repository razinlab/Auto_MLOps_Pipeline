import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import logging

from auth import get_data_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_historical_data(tickers, start_date=None, end_date=None, timeframe=TimeFrame.Day):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    logger.info(f"Fetching historical data for {tickers} from {start_date} to {end_date}")

    client = get_data_client()

    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=timeframe,
        start=start,
        end=end
    )

    try:
        bars = client.get_stock_bars(request_params)

        data_dict = {}
        for ticker in tickers:
            if ticker in bars.data:
                bar_list = []
                for bar in bars.data[ticker]:
                    bar_list.append({
                        'date': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'trade_count':bar.trade_count,
                        'symbol':bar.symbol,
                        'vwap':bar.vwap
                    })

                df = pd.DataFrame(bar_list)

                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

                data_dict[ticker] = df
            else:
                logger.warning(f"No data found for {ticker}")
        
        return data_dict
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise
    
def save_to_parquet(data_dict, directory='data'):
    os.makedirs(directory, exist_ok=True)
    
    for ticker, df in data_dict.items():
        file_path = os.path.join(directory, f"{ticker}.parquet")
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved data for {ticker} to {file_path}")

def load_from_parquet(ticker, directory='data'):
    file_path = os.path.join(directory, f"{ticker}.parquet")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"No data file found for {ticker}")
    
    df = pd.read_parquet(file_path)

    df['date'] = pd.to_datetime(df['date'])
    
    return df

if __name__ == "__main__":
    tickers = ['SPY', 'AAPL', 'TSLA', 'MSFT', 'AMZN']
    
    # Fetch 5 years of historical data
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = fetch_historical_data(tickers, start_date, end_date)
        save_to_parquet(data)
        
        # Print summary of the data
        for ticker in tickers:
            try:
                df = load_from_parquet(ticker)
                print(f"{ticker}: {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
            except FileNotFoundError:
                print(f"No data file found for {ticker}")
    
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")