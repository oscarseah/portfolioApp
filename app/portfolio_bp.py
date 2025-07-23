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
from .portfolio import get_stock_data, filter_stocks, calculate_efficient_frontier, calculate_steepest_descent

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
    try:
        if value is None:
            return ""
        return f"{value:,.{decimals}f}"
    except:
        return value

@bp.route('/')
def home():
    return redirect(url_for('portfolio_bp.show_optimization_form'))

@bp.route('/optimize', methods=['GET'])
def show_optimization_form():
    """Show the optimization form"""
    return render_template('optimize_form.html', error=None)

@bp.route('/optimize', methods=['POST'])
def process_optimization():
    """Process optimization form submission"""
    try:
        capital = float(request.form.get('capital'))
        strategy = request.form.get('strategy')
        
        if capital < 1000:
            raise ValueError("Minimum investment is RM 1,000")
        if strategy not in ['conservative', 'balanced', 'aggressive']:
            raise ValueError("Invalid strategy selected")
        
        global filtered_in
        if not filtered_in:
            initialize_stock_data()
        
        stocks_df = pd.DataFrame(filtered_in)
        
        # Calculate efficient frontier and steepest descent
        frontier_data = calculate_efficient_frontier(stocks_df)
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
        
        # Create allocation dictionaries
        stock_names = stocks_df['Stock'].values
        for port in [low_risk, medium_risk, high_risk]:
            port['allocation'] = {
                name: f"{w*100:.1f}%" 
                for name, w in zip(stock_names, port['weights'])
            }
        
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
        
        # Prepare allocation details
        allocation_details = []
        for stock, alloc in selected_port['allocation'].items():
            percentage = float(alloc.strip('%'))
            allocation_details.append({
                'stock': stock,
                'allocation': alloc,
                'amount': capital * percentage / 100
            })
        
        return render_template('optimize_result.html',
                               capital=capital,
                               strategy=strategy,
                               allocation_details=allocation_details,
                               expected_return=selected_port['return'],
                               expected_risk=selected_port['risk'],
                               plot_url=plot_url)
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        return render_template('optimize_form.html', error=str(e))