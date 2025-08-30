# portfolio_bp.py
from flask import Blueprint, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import logging
import numpy as np
from .portfolio import (
    get_stock_data,
    filter_stocks,
    calculate_efficient_frontier,
    calculate_steepest_descent,
    refine_efficient_frontier,
    calculate_board_lots,
    recompute_metrics,
)

SNAPSHOT_PATH = 'data/processed/portfolio_snapshot.json'
LATEST_PRICE_FILE = 'data/processed/stock analysis 2025 FTSE 100 index.xlsx'
# LATEST_PRICE_FILE = 'data/processed/stock analysis 2025 KLCI 30 index.xlsx'

# Create blueprint
bp = Blueprint('portfolio_bp', __name__)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize stock data
all_stocks = None
filtered_in = None
filtered_out = None

def initialize_stock_data():
    # Load stock data and apply initial filtering

    global all_stocks, filtered_in, filtered_out
    try:
        all_stocks = get_stock_data() or []
        filtered_in, filtered_out = filter_stocks(all_stocks) if all_stocks else ([], [])
        logger.info(f"Initialized {len(all_stocks)} stocks")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        all_stocks, filtered_in, filtered_out = [], [], []
        raise

initialize_stock_data()

# Add template filter to blueprint
@bp.app_template_filter('number_format')
def number_format_filter(value, decimals=0):
    # Format numbers with commas and optional decimals
    try:
        if value is None:
            return ""
        return f"{value:,.{decimals}f}"
    except:
        return value

@bp.route('/')
def home():
    # Redirect to the optimization form
    return redirect(url_for('portfolio_bp.show_optimization_form'))

@bp.route('/optimize', methods=['GET'])
def show_optimization_form():
    # Show the optimization form
    return render_template('optimize_form.html', error=None)

@bp.route('/optimize', methods=['POST'])
def process_optimization():
    # Process optimization form submission
    try:
        capital = float(request.form.get('capital'))
        strategy = request.form.get('strategy')
        
        if capital < 10000:
            raise ValueError("Minimum investment is RM 10,000")
        if strategy not in ['conservative', 'balanced', 'aggressive']:
            raise ValueError("Invalid strategy selected")
        
        global filtered_in
        if not filtered_in:
            initialize_stock_data()
        
        # Filter based on min buy-in
        eligible_stocks = [s for s in filtered_in if capital >= s.get('min_buy', float('inf'))]

        if not eligible_stocks:
            raise ValueError("No stocks meet the minimum buy-in requirement for the provided capital.")

        stocks_df = pd.DataFrame(eligible_stocks)
        
        # Calculate efficient frontier and steepest descent
        raw_frontier = calculate_efficient_frontier(stocks_df)
        frontier_data = refine_efficient_frontier(raw_frontier, stocks_df)
        if not frontier_data:
            raise ValueError("Could not calculate efficient frontier")
            
        steepest_port = calculate_steepest_descent(stocks_df)
        if not steepest_port:
            raise ValueError("Could not calculate minimum risk portfolio")
        steepest_port = steepest_port[0]
        
        # Sort frontier by risk and select portfolios
        frontier_data.sort(key=lambda x: x['risk'])
        low_risk = frontier_data[0]
        high_risk = frontier_data[-1]
        medium_risk = frontier_data[len(frontier_data)//2]

        # Add type labels
        low_risk['type'] = 'Low Risk'
        medium_risk['type'] = 'Medium Risk'
        high_risk['type'] = 'High Risk'

        # Reconstruct weights from allocation strings for each portfolio
        stock_names = stocks_df['Stock'].values
        for port in [low_risk, medium_risk, high_risk]:
            allocation = port.get('allocation', {})
            weights = []
            for name in stock_names:
                alloc_str = allocation.get(name, '0%')
                alloc = float(alloc_str.strip('%')) / 100
                weights.append(alloc)
            port['weights'] = weights
        
        # Select portfolio based on strategy
        if strategy == 'conservative':
            selected_port = low_risk
        elif strategy == 'aggressive':
            selected_port = high_risk
        else:
            selected_port = medium_risk

        # Create efficient frontier plot
        plt.figure(figsize=(10, 6))
        
        # Plot efficient frontier
        risks = [p['risk'] for p in frontier_data]
        returns = [p['return'] for p in frontier_data]
        plt.plot(risks, returns, 'b-', label='Efficient Frontier')
        
        # Mark selected portfolio
        plt.scatter(selected_port['risk'], selected_port['return'], 
                    color='red', s=100, label='Selected Portfolio')
        
        # Mark steepest descent
        plt.scatter(steepest_port['risk'], steepest_port['return'], 
                    color='green', marker='*', s=150, label='Min Risk (Steepest Descent)')
        
        plt.xlabel('Risk (Semi-Annual)')
        plt.ylabel('Return (Semi-Annual)')
        plt.title(f'Portfolio Optimization - {strategy.capitalize()} Strategy')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # Prepare allocation details using board lot multiples and save snapshot
        allocation_details, invested, leftover = calculate_board_lots(
            selected_port['allocation'], capital, stocks_df, snapshot_path=SNAPSHOT_PATH
        )

        # Recompute portfolio metrics using the actual board-lot purchases.
        actual_risk, grew_capital, semi_return, allocation_details = recompute_metrics(
            allocation_details, invested, stocks_df, snapshot_path=SNAPSHOT_PATH, latest_price_file=LATEST_PRICE_FILE
        )
        allocation_sum = sum(d['amount'] for d in allocation_details) / capital * 100

        return render_template('optimize_result.html',
                                capital=capital,
                                strategy=strategy,
                                allocation_details=allocation_details,
                                grew_capital=grew_capital,
                                semi_annual_return=semi_return,
                                portfolio_risk=actual_risk,
                                plot_url=plot_url,
                                leftover=leftover,
                                allocation_sum=allocation_sum,
                                )
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        return render_template('optimize_form.html', error=str(e))