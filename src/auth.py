import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

def get_trading_client():
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
    
    return TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

def get_data_client():
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
    
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

def verify_auth():
    client = get_trading_client()
    return client.get_account()

if __name__ == "__main__":
    try:
        account = verify_auth()
        print(f"Account ID: {account.id}")
        print(f"Account status: {account.status}")
        print(f"Buying power: ${account.buying_power}")
        print(f"Cash: ${account.cash}")
        print(f"Portfolio value: ${account.portfolio_value}")
    except Exception as e:
        print(f"Authentication failed: {e}")