import numpy as np
import pandas as pd

def get_stock_data():
    """Load stock data with annual return/risk data"""
    try:
        df = pd.read_csv('data/malaysia_stocks.csv') 
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading stocks: {str(e)}")
        return []

def filter_stocks(stocks):
    """
    Filter stocks based on Sharpe ratio and minimum return
    Returns two lists sorted by Sharpe Ratio
    """
    try:
        filtered_in = []
        filtered_out = []
        
        for stock in stocks:
            # Calculate Sharpe ratio (return/risk)
            sharpe_ratio = stock['return'] / stock['risk'] if stock['risk'] > 0 else 0
            stock['sharpe'] = sharpe_ratio  # Add for all stocks
            
            # Apply filters: Sharpe ratio >= 1
            if sharpe_ratio >= 1:
                filtered_in.append(stock)
            else:
                filtered_out.append(stock)
                
        # Sort filtered_in by Sharpe ratio 
        filtered_in = sorted(filtered_in, key=lambda x: x['sharpe'], reverse=True)
 
        filtered_out = sorted(filtered_out, key=lambda x: x['sharpe'], reverse=True)
        
        return filtered_in, filtered_out
    
    except Exception as e:
        print(f"Filtering error: {str(e)}")
        return [], []