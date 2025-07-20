import pandas as pd
import numpy as np
import math
import os

# Set risk-free rate (semi-annual)
SEMI_ANNUAL_RISK_FREE = 0.0145  # 1.45% per semi-annual period

def get_stock_data():
    """Load stock data with annual return/risk data"""
    try:
        # First check if we already have the processed file
        processed_file = 'stock_analysis_with_sharpe.xlsx'
        if os.path.exists(processed_file):
            df = pd.read_excel(processed_file)
        else:
            # If not, generate it from the raw data
            input_file = 'stock_returns_risk.xlsx'
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Required file not found: {input_file}")
            
            results_df = pd.read_excel(input_file)
            results_df = calculate_sharpe_ratios(results_df)
            results_df.to_excel(processed_file, index=False)
            df = results_df
        
        # Convert to list of dictionaries with consistent key names
        stocks = []
        for _, row in df.iterrows():
            stocks.append({
                'Stock': row['Stock'],
                'return': row['Semi-Annual Return'],
                'risk': row['Semi-Annual Risk'],
                'sharpe': row.get('Annualized Sharpe Ratio', np.nan)  # Use get for backward compatibility
            })
        return stocks
    
    except Exception as e:
        print(f"Error loading stocks: {str(e)}")
        return []

def calculate_sharpe_ratios(results_df):
    """Calculate annualized Sharpe ratios for a DataFrame"""
    sharpe_ratios = []
    for _, row in results_df.iterrows():
        avg_return = row['Semi-Annual Return']
        risk = row['Semi-Annual Risk']
        
        # Calculate excess return
        excess_return = avg_return - SEMI_ANNUAL_RISK_FREE
        
        # Handle near-zero risk cases
        if risk > 1e-8:
            # Annualize Sharpe ratio: multiply by sqrt(2) for semi-annual data
            sharpe_ratio = (excess_return / risk) * math.sqrt(2)
        else:
            sharpe_ratio = np.nan
        
        sharpe_ratios.append(sharpe_ratio)
    
    # Add Sharpe ratio to DataFrame
    results_df['Annualized Sharpe Ratio'] = sharpe_ratios
    
    # Round all numeric columns to 4 decimal places
    numeric_cols = ['Semi-Annual Return', 'Semi-Annual Risk', 'Annualized Sharpe Ratio']
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    return results_df

def filter_stocks(stocks):
    """
    Filter stocks based on Sharpe ratio and minimum return
    Returns two lists sorted by Sharpe Ratio
    """
    try:
        filtered_in = []
        filtered_out = []
        
        for stock in stocks:
            # Use pre-calculated Sharpe ratio if available
            sharpe_ratio = stock.get('sharpe')
            
            # Apply filters: Sharpe ratio >= 1
            if sharpe_ratio is not None and sharpe_ratio >= 0.7:
                filtered_in.append(stock)
            else:
                filtered_out.append(stock)
                
        # Sort filtered_in by Sharpe ratio 
        filtered_in = sorted(filtered_in, key=lambda x: x.get('sharpe', 0), reverse=True)
        filtered_out = sorted(filtered_out, key=lambda x: x.get('sharpe', 0), reverse=True)
        
        return filtered_in, filtered_out
    
    except Exception as e:
        print(f"Filtering error: {str(e)}")
        return [], []

# This will run when the module is executed directly
if __name__ == "__main__":
    # If run as script, generate and save the analysis
    try:
        stocks = get_stock_data()
        if stocks:
            # Save processed data
            df = pd.DataFrame(stocks)
            df.to_excel('stock_analysis_with_sharpe.xlsx', index=False)
            
            print(f"Using semi-annual risk-free rate: {SEMI_ANNUAL_RISK_FREE:.4f}")
            print(f"Saved processed data to 'stock_analysis_with_sharpe.xlsx'")
            
            # Show top performers
            filtered_in, _ = filter_stocks(stocks)
            top_stocks = pd.DataFrame(filtered_in).head()
            print("\nTop performing stocks by Sharpe ratio:")
            print(top_stocks[['Stock', 'return', 'risk', 'sharpe']].to_string(index=False))
    except Exception as e:
        print(f"Runtime error: {str(e)}")