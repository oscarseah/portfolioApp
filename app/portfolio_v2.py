import pandas as pd
import numpy as np
import math
import os
from scipy.optimize import minimize

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

# Add these optimization functions at the bottom of portfolio.py
def calculate_efficient_frontier(stocks_df):
    """Calculate efficient frontier portfolios from filtered stocks DataFrame"""
    try:
        # Input validation
        if len(stocks_df) < 2:
            raise ValueError("Select at least 2 different stocks")

        # Extract parameters from DataFrame
        expected_returns = stocks_df['return'].values
        risks = stocks_df['risk'].values
        n_stocks = len(stocks_df)
        stock_names = stocks_df['Stock'].values

        # Construct covariance matrix assuming 65% correlation between stocks
        correlation = 0.65  # Stocks in same market typically correlate
        cov_matrix = np.outer(risks, risks) * correlation
        np.fill_diagonal(cov_matrix, risks**2)  # Set diagonal to variance

        # Portfolio functions
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)
        
        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Set optimization constraints
        bounds = [(0.01, 0.9) for _ in range(n_stocks)]  # 1%-90% per stock
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum to 100%

        # Target returns range
        min_return = np.min(expected_returns) * 0.9  # 10% below minimum
        max_return = np.max(expected_returns) * 1.1  # 10% above maximum
        target_returns = np.linspace(min_return, max_return, 30)  # 30 points

        # Storage for optimized portfolios
        efficient_portfolios = []

        # Optimize for each target return
        for target in target_returns:
            constraints.append({'type': 'eq', 'fun': lambda x, t=target: portfolio_return(x) - t})
            
            result = minimize(
                portfolio_risk,
                x0=np.ones(n_stocks)/n_stocks,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                efficient_portfolios.append({
                    'return': portfolio_return(result.x),
                    'risk': portfolio_risk(result.x),
                    'weights': result.x
                })
            constraints.pop()  # Remove temporary constraint

        # Sort portfolios by risk
        efficient_portfolios.sort(key=lambda x: x['risk'])

        return efficient_portfolios

    except Exception as e:
        print(f"Efficient Frontier error: {str(e)}")
        return []

def calculate_steepest_descent(stocks_df, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """Calculate minimum risk portfolio using gradient descent"""
    try:
        # Input validation
        if len(stocks_df) < 2:
            raise ValueError("Select at least 2 different stocks")

        # Extract parameters
        expected_returns = stocks_df['return'].values
        risks = stocks_df['risk'].values
        n_stocks = len(stocks_df)
        stock_names = stocks_df['Stock'].values

        # Construct covariance matrix
        correlation = 0.65
        cov_matrix = np.outer(risks, risks) * correlation
        np.fill_diagonal(cov_matrix, risks**2)

        # Define objective and gradient
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def gradient(weights):
            port_risk = objective(weights)
            if port_risk < 1e-10:
                return np.zeros_like(weights)
            return np.dot(cov_matrix, weights) / port_risk

        # Initialize weights
        weights = np.ones(n_stocks) / n_stocks
        prev_risk = float('inf')
        
        # Gradient descent loop
        for _ in range(max_iter):
            grad = gradient(weights)
            weights -= learning_rate * grad
            
            # Project onto simplex
            weights = np.clip(weights, 0.01, 0.9)
            weights /= weights.sum()
            
            current_risk = objective(weights)
            if abs(prev_risk - current_risk) < tolerance:
                break
            prev_risk = current_risk
    
        # Return optimized portfolio
        allocations = {name: f"{w*100:.1f}%" for name, w in zip(stock_names, weights)}
        return [{
            'return': np.dot(weights, expected_returns),
            'risk': current_risk,
            'allocation': allocations,
            'type': 'Minimum Risk'
        }]

    except Exception as e:
        print(f"Steepest Descent error: {str(e)}")
        return []