import pandas as pd
import numpy as np
import math
import os
from scipy.optimize import minimize

def get_stock_data():
    """Load stock data with semi-annual return/risk/sharpe and min buy-in"""
    try:
        processed_file = 'stock_analysis_results_with_sharpe.xlsx'
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Required file not found: {processed_file}")

        df = pd.read_excel(processed_file)
        required_cols = ['Stock', 'Min Buy In (RM)', 'Semi-Annual Return', 'Semi-Annual Risk', 'Sharpe Ratio']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        stocks = []
        for _, row in df.iterrows():
            stocks.append({
                'Stock': row['Stock'],
                'return': row['Semi-Annual Return'],
                'risk': row['Semi-Annual Risk'],
                'sharpe': row['Sharpe Ratio'],
                'min_buy': row['Min Buy In (RM)']
            })
        return stocks

    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return []

def calculate_board_lots(allocation, capital, stocks_df):
    details = []
    invested = 0.0

    for stock, alloc_str in allocation.items():
        try:
            percentage = float(alloc_str.strip('%'))
        except (ValueError, AttributeError):
            continue

        target_amount = capital * percentage / 100
        row = stocks_df.loc[stocks_df['Stock'] == stock]
        if row.empty:
            continue
        min_buy = float(row['min_buy'].iloc[0])

        board_lots = int(target_amount // min_buy)
        if board_lots < 1:
            continue

        amount = board_lots * min_buy
        units = board_lots * 100
        invested += amount

        details.append({
            'stock': stock,
            'allocation': alloc_str,
            'amount': amount,
            'units': units,
        })

    leftover = capital - invested
    return details, invested, leftover

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

# Portfolio optimization functions
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
            # Create a copy of constraints to avoid mutation issues
            current_constraints = constraints.copy()
            current_constraints.append({'type': 'eq', 'fun': lambda x, t=target: portfolio_return(x) - t})
            
            result = minimize(
                portfolio_risk,
                x0=np.ones(n_stocks)/n_stocks,
                method='SLSQP',
                bounds=bounds,
                constraints=current_constraints
            )
            
            if result.success:
                efficient_portfolios.append({
                    'return': portfolio_return(result.x),
                    'risk': portfolio_risk(result.x),
                    'weights': result.x,
                    'target_return': target
                })
        
        # Sort portfolios by risk
        efficient_portfolios.sort(key=lambda x: x['risk'])

        return efficient_portfolios

    except Exception as e:
        print(f"Efficient Frontier error: {str(e)}")
        return []

def refine_efficient_frontier(efficient_portfolios, stocks_df, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    """Refine efficient frontier portfolios using steepest descent"""
    try:
        # Input validation
        if not efficient_portfolios:
            raise ValueError("Efficient frontier is empty")
            
        # Extract parameters from stock data
        expected_returns = stocks_df['return'].values
        risks = stocks_df['risk'].values
        stock_names = stocks_df['Stock'].values
        
        # Construct covariance matrix
        correlation = 0.65
        cov_matrix = np.outer(risks, risks) * correlation
        np.fill_diagonal(cov_matrix, risks**2)
        
        # Storage for refined portfolios
        refined_portfolios = []
        
        for portfolio in efficient_portfolios:
            # Get original weights and target return
            weights = portfolio['weights'].copy()
            target_return = portfolio['target_return']
            
            # Define objective and gradient with return constraint
            def objective(w):
                return np.sqrt(w.T @ cov_matrix @ w)
                
            def gradient(w):
                port_risk = objective(w)
                if port_risk < 1e-10:
                    return np.zeros_like(w)
                return (cov_matrix @ w) / port_risk
                
            def return_constraint(w):
                return np.dot(w, expected_returns) - target_return
                
            # Gradient descent with return constraint
            prev_risk = objective(weights)
            for _ in range(max_iter):
                # Compute gradient
                grad = gradient(weights)
                
                # Update weights
                new_weights = weights - learning_rate * grad
                
                # Project onto constraints
                new_weights = np.clip(new_weights, 0.01, 0.9)
                new_weights /= new_weights.sum()
                
                # Adjust to maintain target return
                if abs(return_constraint(new_weights)) > 0.001:
                    # Simple projection to maintain return
                    current_return = np.dot(new_weights, expected_returns)
                    adjustment = target_return - current_return
                    return_diffs = expected_returns - np.mean(expected_returns)
                    new_weights += adjustment * return_diffs / (return_diffs @ return_diffs)
                    new_weights = np.clip(new_weights, 0.01, 0.9)
                    new_weights /= new_weights.sum()
                
                # Check convergence
                current_risk = objective(new_weights)
                if abs(prev_risk - current_risk) < tolerance:
                    weights = new_weights
                    break
                    
                weights = new_weights
                prev_risk = current_risk
            
            # Store refined portfolio
            allocations = {name: f"{w*100:.1f}%" for name, w in zip(stock_names, weights)}
            refined_portfolios.append({
                'return': np.dot(weights, expected_returns),
                'risk': current_risk,
                'target_return': target_return,
                'allocation': allocations,
                'original_risk': portfolio['risk'],
                'improvement': portfolio['risk'] - current_risk
            })
        
        return refined_portfolios

    except Exception as e:
        print(f"EF Refinement error: {str(e)}")
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