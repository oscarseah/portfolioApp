import pandas as pd
import numpy as np
import math
import os
import json
from scipy.optimize import minimize

def get_stock_data():
    # Load stock data and convert semi-annual returns and risk to annual figures
    try:
        
        # processed_file = 'data/processed/stock analysis 2021-2024 KLCI 30 index.xlsx'
        processed_file = 'data/processed/stock analysis 2021-2024 FTSE 100 index.xlsx'
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Required file not found: {processed_file}")

        df = pd.read_excel(processed_file)
        required_cols = ['Stock', 'Min Buy In (RM)', 'Semi-Annual Return', 'Semi-Annual Risk', 'Sharpe Ratio']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        stocks = []
        for _, row in df.iterrows():
            sa_return = row['Semi-Annual Return']
            annual_return = (1 + sa_return) ** 2 - 1
            sa_risk = row['Semi-Annual Risk']
            annual_risk = sa_risk * math.sqrt(2)
            stocks.append({
                'Stock': row['Stock'],
                'return': annual_return,
                'risk': annual_risk,
                'sharpe': row['Sharpe Ratio'],
                'min_buy': row['Min Buy In (RM)']
            })
        return stocks

    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return []

def calculate_board_lots(allocation, capital, stocks_df, snapshot_path=None):
    """Calculate board-lot purchases for each stock allocation.

    If ``snapshot_path`` is provided the invested capital and holdings are
    persisted to that JSON file for later performance calculations.
    """

    details = []
    invested = 0.0

    min_buy_lookup = stocks_df.set_index('Stock')['min_buy'].to_dict()

    # First pass: allocate based on requested percentages
    for stock, alloc_str in allocation.items():
        try:
            percentage = float(alloc_str.strip('%'))
        except (ValueError, AttributeError):
            continue

        target_amount = capital * percentage / 100
        min_buy = float(min_buy_lookup.get(stock, 0))
        if min_buy <= 0:
            continue

        board_lots = int(target_amount // min_buy)
        if board_lots < 1:
            continue

        amount = board_lots * min_buy
        units = board_lots * 100
        invested += amount

        realized_pct = amount / capital * 100 if capital else 0

        details.append({
            'stock': stock,
            'allocation': f"{realized_pct:.2f}%",
            'amount': amount,
            'units': units,
        })

    leftover = capital - invested

    if snapshot_path:
        save_portfolio_snapshot(details, invested, snapshot_path)

    return details, invested, leftover

def save_portfolio_snapshot(allocation_details, invested, path):
    """Persist portfolio allocation and invested capital to JSON."""
    try:
        snapshot = {
            'invested_capital': invested,
            'holdings': [
                {
                    'stock': d['stock'],
                    'units': d['units']
                } for d in allocation_details
            ]
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(snapshot, f)
    except Exception as e:
        print(f"Error saving portfolio snapshot: {e}")


def calculate_annual_return(snapshot_path, latest_price_file):
    # Calculate current portfolio value and convert the semi-annual growth to annual return.
    try:
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

        # Load the snapshot saved at purchase time.
        # It records the amount invested and the number of units purchased for each stock.
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)

        invested = snapshot.get('invested_capital', 0)
        holdings = snapshot.get('holdings', [])
        if invested <= 0 or not holdings:
            return 0.0, 0.0

        # Read the latest market prices.
        df_latest = pd.read_excel(latest_price_file)
        required_cols = {'Stock', 'Latest Price'}
        if not required_cols.issubset(df_latest.columns):
            raise ValueError("Latest price file missing required columns")

        price_lookup = df_latest.set_index('Stock')['Latest Price'].to_dict()

        # Revalue the holdings at current market prices.
        grew_capital = 0.0
        for h in holdings:
            stock = h['stock']
            units = h['units']
            price = price_lookup.get(stock)
            if price is not None:
                grew_capital += units * price

        semi_annual_return = (grew_capital - invested) / invested if invested else 0.0
        annual_return = (1 + semi_annual_return) ** 2 - 1
        return grew_capital, annual_return

    except Exception as e:
        print(f"Error calculating annual return: {e}")
        return 0.0, 0.0

def recompute_metrics(allocation_details, invested, stocks_df, snapshot_path=None, latest_price_file=None):
    """Recompute portfolio risk and performance based on actual holdings."""


    if invested <= 0 or stocks_df.empty:
        return 0.0, 0.0, 0.0, allocation_details

    names = stocks_df['Stock'].tolist()

    # Prepare weight vector based on actual invested amounts. 
    # Each weight is the fraction of total invested capital allocated to that stock.
    weights = []
    for name in names:
        detail = next((d for d in allocation_details if d['stock'] == name), None)
        w = detail['amount'] / invested if detail else 0.0
        weights.append(w)
        if detail is not None:
            # Update allocation to reflect actual weight achieved after rounding to board-lot multiples.
            detail['allocation'] = f"{w*100:.2f}%"

    # Risk is derived from historical data
    # the covariance matrix built from individual stock risks and an assumed constant correlation.
    _, risks, _, cov_matrix = build_portfolio_params(stocks_df)

    # Portfolio risk is the standard deviation of returns, computed as
    # sqrt(w^T Σ w) where Σ is the covariance matrix and w is the weight vector.
    w_vec = np.array(weights)
    port_risk = float(np.sqrt(w_vec.T @ cov_matrix @ w_vec))

    # Optionally compute current portfolio value and realized annual return
    # if a snapshot and latest prices are provided.
    grew_capital = 0.0
    annual_return = 0.0
    if snapshot_path and latest_price_file:
        grew_capital, annual_return = calculate_annual_return(snapshot_path, latest_price_file)

    return port_risk, grew_capital, annual_return, allocation_details

def filter_stocks(stocks):
    # Filter stocks based on Sharpe ratio and minimum return
    # Returns two lists sorted by Sharpe Ratio
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

# Helper to build common portfolio parameters
def build_portfolio_params(stocks_df, correlation=0.65):
    expected_returns = stocks_df['return'].values
    risks = stocks_df['risk'].values
    stock_names = stocks_df['Stock'].values
    cov_matrix = np.outer(risks, risks) * correlation
    np.fill_diagonal(cov_matrix, risks**2)
    return expected_returns, risks, stock_names, cov_matrix


def calculate_efficient_frontier(stocks_df):
    # Calculate efficient frontier portfolios from filtered stocks DataFrame
    try:
        if len(stocks_df) < 2:
            raise ValueError("Select at least 2 different stocks")

        # Common setup
        expected_returns, risks, stock_names, cov_matrix = build_portfolio_params(stocks_df)
        n_stocks = len(stocks_df)

        # Portfolio helper functions
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        def portfolio_risk(weights):
            # Standard deviation of portfolio returns for a given weight vector
            # calculated via sqrt(w^T Σ w)
            return np.sqrt(weights.T @ cov_matrix @ weights)

        # Constraints
        bounds = [(0.01, 0.9)] * n_stocks
        base_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Target returns range
        min_return = expected_returns.min() * 0.9
        max_return = expected_returns.max() * 1.1
        target_returns = np.linspace(min_return, max_return, 50)

        efficient_portfolios = []

        for t in target_returns:
            constraints = [base_constraint,
                           {'type': 'eq', 'fun': lambda x, target=t: portfolio_return(x) - target}]
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
                    'weights': result.x,
                    'target_return': t
                })

        efficient_portfolios.sort(key=lambda x: x['risk'])
        return efficient_portfolios

    except Exception as e:
        print(f"Efficient Frontier error: {str(e)}")
        return []


def refine_efficient_frontier(efficient_portfolios, stocks_df, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    # Refine efficient frontier portfolios using steepest descent
    try:
        if not efficient_portfolios:
            raise ValueError("Efficient frontier is empty")

        expected_returns, risks, stock_names, cov_matrix = build_portfolio_params(stocks_df)
        refined_portfolios = []

        for port in efficient_portfolios:
            weights = port['weights'].copy()
            target_return = port['target_return']

            def objective(w):
                return np.sqrt(w.T @ cov_matrix @ w)

            def gradient(w):
                r = objective(w)
                # Gradient of sqrt(w^T Σ w) with respect to w is (Σ w) / r
                # where r is the current portfolio risk.
                return np.zeros_like(w) if r < 1e-10 else (cov_matrix @ w) / r

            def return_constraint(w):
                return np.dot(w, expected_returns) - target_return

            prev_risk = objective(weights)
            for _ in range(max_iter):
                grad = gradient(weights)
                new_w = weights - learning_rate * grad
                new_w = np.clip(new_w, 0.01, 0.9)
                new_w /= new_w.sum()

                # Maintain return
                if abs(return_constraint(new_w)) > 0.001:
                    current_ret = new_w @ expected_returns
                    adj = target_return - current_ret
                    diffs = expected_returns - expected_returns.mean()
                    new_w += adj * diffs / (diffs @ diffs)
                    new_w = np.clip(new_w, 0.01, 0.9)
                    new_w /= new_w.sum()

                curr_risk = objective(new_w)
                if abs(prev_risk - curr_risk) < tolerance:
                    weights = new_w
                    break

                weights = new_w
                prev_risk = curr_risk

            allocs = {name: f"{w*100:.2f}%" for name, w in zip(stock_names, weights)}
            refined_portfolios.append({
                'return': weights @ expected_returns,
                'risk': curr_risk,
                'target_return': target_return,
                'allocation': allocs,
                'original_risk': port['risk'],
                'improvement': port['risk'] - curr_risk
            })

        return refined_portfolios

    except Exception as e:
        print(f"EF Refinement error: {str(e)}")
        return []

def calculate_steepest_descent(stocks_df, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
    # Calculate minimum risk portfolio using gradient descent
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
            # Compute portfolio risk for given weights using the covariance
            # matrix: sqrt(w^T Σ w)
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def gradient(weights):
            # Gradient of the risk objective sqrt(w^T Σ w).
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
        allocations = {name: f"{w*100:.2f}%" for name, w in zip(stock_names, weights)}
        return [{
            'return': np.dot(weights, expected_returns),
            'risk': current_risk,
            'allocation': allocations,
            'type': 'Minimum Risk'
        }]

    except Exception as e:
        print(f"Steepest Descent error: {str(e)}")
        return []