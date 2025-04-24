import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os

# Create a directory for storing data
os.makedirs('data', exist_ok=True)

def get_nifty50_data():
    """
    Collect Nifty 50 data using YahooFinance API
    """
    print("Collecting Nifty 50 data...")
    
    # Using the data API client
    client = ApiClient()
    
    # Get Nifty 50 data (^NSEI is the symbol for Nifty 50)
    nifty_data = client.call_api('YahooFinance/get_stock_chart', query={
        'symbol': '^NSEI',
        'region': 'IN',
        'interval': '1d',
        'range': '1y',
        'includeAdjustedClose': True
    })
    
    # Extract the time series data
    timestamps = nifty_data['chart']['result'][0]['timestamp']
    indicators = nifty_data['chart']['result'][0]['indicators']
    
    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Extract OHLCV data
    quote = indicators['quote'][0]
    opens = quote['open']
    highs = quote['high']
    lows = quote['low']
    closes = quote['close']
    volumes = quote['volume']
    
    # Extract adjusted close if available
    adj_closes = indicators['adjclose'][0]['adjclose'] if 'adjclose' in indicators else closes
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Adj Close': adj_closes,
        'Volume': volumes
    })
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    # Save to CSV
    df.to_csv('data/nifty50_data.csv')
    print(f"Nifty 50 data saved to data/nifty50_data.csv")
    
    return df

def get_nifty50_constituents():
    """
    Get list of Nifty 50 constituent stocks
    """
    print("Collecting Nifty 50 constituent stocks...")
    
    # Using yfinance as a backup to get constituent data
    # This is a list of major Nifty 50 constituents
    nifty_constituents = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'BAJAJFINSV.NS', 'WIPRO.NS', 'HCLTECH.NS',
        'ADANIENT.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS'
    ]
    
    # Create an empty DataFrame to store all constituent data
    all_data = pd.DataFrame()
    
    # Get data for each constituent
    for symbol in nifty_constituents:
        try:
            # Using yfinance directly for individual stocks
            stock_data = yf.download(symbol, period='1y', interval='1d')
            stock_data['Symbol'] = symbol
            
            # Append to the main DataFrame
            if all_data.empty:
                all_data = stock_data
            else:
                all_data = pd.concat([all_data, stock_data])
                
            print(f"Collected data for {symbol}")
        except Exception as e:
            print(f"Error collecting data for {symbol}: {e}")
    
    # Save to CSV
    all_data.to_csv('data/nifty50_constituents.csv')
    print(f"Nifty 50 constituents data saved to data/nifty50_constituents.csv")
    
    return all_data

if __name__ == "__main__":
    # Get Nifty 50 index data
    nifty_data = get_nifty50_data()
    
    # Get Nifty 50 constituent stocks data
    constituents_data = get_nifty50_constituents()
    
    print("Data collection completed successfully!")
