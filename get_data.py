import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

# List of Bursa Malaysia tickers
tickers = [
    "1015.KL", # AMMB Holdings Berhad
    "1023.KL", # CIMB Group Holdings Berhad
    "1066.KL",  # RHB Malaysia
    "1155.KL",  # Maybank
    "1295.KL",  # Public Bank
    "2089.KL", # United Plantations Berhad
    "4197.KL", # Sime Darby Berhad
    "4863.KL", # Telekom Malaysia Berhad
    "5024.KL", # Hup Seng
    "5211.KL", # Sunway Berhad
    "5347.KL",  # Tenaga Nasional
    "5398.KL", # Gamuda Berhad
    "6012.KL", # Maxis Berhad
    "6033.KL",   # Petronas Malaysia
    "6947.KL", # CelcomDigi Berhad
]

def process_stock(ticker):
    try:
        print(f"\nProcessing {ticker}...")
        
        # Set date range (3 years back from current date)
        end_date = datetime(2025, 4, 4)  # Your current date
        start_date = end_date - pd.DateOffset(years=3)
        
        # Download data with Close prices
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1mo",
            progress=False
        )
        
        # Check if data exists
        if data.empty:
            print(f"No data found for {ticker}")
            return
            
        # Verify Close price exists
        if 'Close' not in data.columns:
            print(f"Missing Close price for {ticker}")
            return
            
        # Calculate monthly returns using Close price
        data['Monthly Return'] = data['Close'].pct_change()
        data = data.dropna()
        
        # Check sufficient data
        if len(data) < 6:
            print(f"Insufficient data points ({len(data)}) for {ticker}")
            return
            
        # Calculate annualized metrics
        annual_return = data['Monthly Return'].mean() * 12
        annual_risk = data['Monthly Return'].std() * np.sqrt(12)
        
        # Save results
        data[['Close', 'Monthly Return']].to_csv(f"data/individual/{ticker}_historical.csv")
        pd.DataFrame({
            'Ticker': [ticker],
            'Start Date': [data.index[0].strftime('%Y-%m-%d')],
            'End Date': [data.index[-1].strftime('%Y-%m-%d')],
            'Annual Return': [annual_return],
            'Annual Risk': [annual_risk],
            'Data Points': [len(data)]
        }).to_csv(f"data/individual/{ticker}_metrics.csv", index=False)
        
        print(f"Success: Processed {ticker} with {len(data)} months data")
        print(f"First date: {data.index[0].date()}")
        print(f"Last date: {data.index[-1].date()}")

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

# Process all tickers
for ticker in tickers:
    process_stock(ticker)

print("\nProcessing complete!")