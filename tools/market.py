import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_DIR = "cache/market_data"

class MarketData:
    def __init__(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def get_daily_data(self, symbol: str, years: int = 1) -> pd.DataFrame:
        \"\"\"Fetch daily OHLCV data with simple file-based caching\"\"\"
        cache_file = f"{CACHE_DIR}/{symbol}_daily.csv"
        
        # Use cache if it was updated in the last 24 hours
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - mtime < 86400:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Save to cache
            df.to_csv(cache_file)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> float:
        \"\"\"Get most recent closing price\"\"\"
        df = self.get_daily_data(symbol)
        if not df.empty:
            return float(df['Close'].iloc[-1])
        return 0.0
