import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_stock_data():
    """Load stock data with annual return/risk data"""
    try:
        df = pd.read_csv('data/malaysia_stocks.csv') 
        return df.to_dict('records') # Convert DataFrame to list of dictionaries for easier processing
    except Exception as e:
        print(f"Error loading stocks: {str(e)}")
        return []

def calculate_efficient_frontier(selected_stocks):
    try:
        # Data loading and validation
        df = pd.read_csv('data/malaysia_stocks.csv')
        df = df[df['symbol'].isin(selected_stocks)].copy()
        
        # Input validation
        if len(df) < 2:
            raise ValueError("Select at least 2 different stocks")

        # Extract parameters from CSV
        expected_returns = df['return'].values
        risks = df['risk'].values
        n_stocks = len(df)

        # Construct covariance matrix assuming 65% correlation between stocks
        correlation = 0.65  # Stocks in same market typically correlate
        cov_matrix = np.outer(risks, risks) * correlation # Outer product of risks scaled by correlation
        np.fill_diagonal(cov_matrix, risks**2)  # Set diagonal to variance (risk squared)

        # Portfolio functions
        def portfolio_return(weights):
            # Calculate portfolio's expected return
            return np.dot(weights, expected_returns)
        
        def portfolio_risk(weights):
            # Calculate portfolio risk (standard deviation)
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Set optimization constraints
        bounds = [(0.01, 0.9) for _ in range(n_stocks)]  # 1%-90% per stock (to avoid extreme distribution)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # Sum to 100%
        ]

        # Target returns range
        min_return = np.min(expected_returns) * 0.9 # 10% below minimum
        max_return = np.max(expected_returns) * 1.1 # 10% above maximum
        target_returns = np.linspace(min_return, max_return, 30) # Generate 30 evenly spaced target returns

        # Storage for optimized portfolios
        efficient_portfolios = []

        # Optimize for each target return
        for target in target_returns:
            # Add target return constraint
            constraints.append({'type': 'eq', 'fun': lambda x, t=target: portfolio_return(x) - t})
            
            # Perform quadratic optimization
            result = minimize(
                portfolio_risk,
                x0=np.ones(n_stocks)/n_stocks,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Store successful optimizations
            if result.success:
                efficient_portfolios.append({
                    'return': portfolio_return(result.x),
                    'risk': portfolio_risk(result.x),
                    'weights': result.x
                })
            constraints.pop() # Remove temporary target constraint for next iteration

        # Sort portfolios by risk (standard deviation)
        efficient_portfolios.sort(key=lambda x: x['risk'])

        # Select low, medium, and high risk portfolios
        selected_indices = [0, len(efficient_portfolios)//2, -1]
        suggestions = []
        for idx in selected_indices:
            if idx < len(efficient_portfolios):
                port = efficient_portfolios[idx]
                suggestions.append({
                    'return': port['return'],
                    'risk': port['risk'],
                    'allocation': {sym: f"{w*100:.1f}%" 
                                  for sym, w in zip(df['symbol'], port['weights'])}
                })
        # Return top 3 portfolio suggestions
        return suggestions[:3]

    # Error handling
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        raise ValueError(f"Portfolio optimization failed: {str(e)}")
    
def calculate_steepest_descent(selected_stocks, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    try:
        # Load data using same method as efficient frontier
        df = pd.read_csv('data/malaysia_stocks.csv')
        df = df[df['symbol'].isin(selected_stocks)].copy()
        
        if len(df) < 2:
            raise ValueError("Select at least 2 different stocks")

        # Same covariance setup as before
        expected_returns = df['return'].values
        risks = df['risk'].values
        n_stocks = len(df)
        correlation = 0.65
        cov_matrix = np.outer(risks, risks) * correlation
        np.fill_diagonal(cov_matrix, risks**2)

        # Define objective and gradient
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def gradient(weights):
            port_risk = objective(weights)
            if port_risk < 1e-10:  # Avoid division by zero
                return np.zeros_like(weights)
            return np.dot(cov_matrix, weights) / port_risk

        # Initialize weights and constraints
        weights = np.ones(n_stocks) / n_stocks
        prev_risk = float('inf')
        
        # Gradient descent loop
        for _ in range(max_iter):
            grad = gradient(weights)
            weights -= learning_rate * grad
            
            # Project onto simplex (sum to 1, non-negative)
            weights = np.clip(weights, 0.01, 0.9)
            weights /= weights.sum()
            
            current_risk = objective(weights)
            if abs(prev_risk - current_risk) < tolerance:
                break
            prev_risk = current_risk
    
        # Return single optimized portfolio
        return [{
            'return': np.dot(weights, expected_returns),
            'risk': current_risk,
            'allocation': {sym: f"{w*100:.1f}%" for sym, w in zip(df['symbol'], weights)}
        }]

    except Exception as e:
        print(f"Steepest Descent error: {str(e)}")
        raise ValueError(f"Steepest Descent optimization failed: {str(e)}")