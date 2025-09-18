# portfolio_bp.py
from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.io as pio
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

# Fixed deposit annual interest rate for Maybank
MAYBANK_FD_RATE = 0.022 

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

        # Prepare allocation details using board lot multiples and save snapshot
        allocation_details, invested, leftover = calculate_board_lots(
            selected_port['allocation'], capital, stocks_df, snapshot_path=SNAPSHOT_PATH
        )

        # Recompute portfolio metrics using the actual board-lot purchases.
        actual_risk, stock_value, _, allocation_details = recompute_metrics(
            allocation_details, invested, stocks_df, snapshot_path=SNAPSHOT_PATH, latest_price_file=LATEST_PRICE_FILE
        )
        allocation_sum = sum(d['amount'] for d in allocation_details) / capital * 100

        # Create efficient frontier plot
        risks = [p['risk'] for p in frontier_data]
        returns = [p['return'] for p in frontier_data]

        frontier_trace = go.Scatter(
            x=risks, 
            y=returns, 
            mode='lines',
            name='Efficient Frontier',
            hoverinfo='skip'
        )

        # Individual stocks as scatter points with hover showing stock name
        # Show only stocks that were actually purchased (after board-lot rounding)
        included_names = [d['stock'] for d in allocation_details]
        included_df = stocks_df[stocks_df['Stock'].isin(included_names)]
        stock_trace = go.Scatter(
            x=included_df['risk'],
            y=included_df['return'],
            mode='markers',
            name='Portfolio Stocks',
            text=included_df['Stock'],
            hovertemplate='%{text}<extra></extra>',
            marker=dict(size=8, color='gray')
        )

        selected_trace = go.Scatter(
            x=[selected_port['risk']],
            y=[selected_port['return']],
            mode='markers',
            name='Selected Portfolio',
            marker=dict(color='red', size=12),
            hovertemplate='Selected Portfolio<extra></extra>',
        )

        steepest_trace = go.Scatter(
            x=[steepest_port['risk']],
            y=[steepest_port['return']],
            mode='markers',
            name='Min Risk (Steepest Descent)',
            marker=dict(color='green', size=15, symbol='star'),
            hovertemplate='Min Risk (Steepest Descent)<extra></extra>',
        )

        fig = go.Figure(data=[frontier_trace, stock_trace,
                               selected_trace, steepest_trace])
        fig.update_layout(
            xaxis_title='Risk (Annual)',
            yaxis_title='Return (Annual)',
            title=f'Strategy Allocation Results',
            template='plotly_white',
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            hoverlabel=dict(bgcolor="white"),
        )

        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', default_width='100%', default_height='500px')

        # Combine stock growth with fixed-deposit interest for unused capital.
        # Only amounts above RM500 qualify for fixed deposit interest.
        fd_interest = leftover * MAYBANK_FD_RATE / 2 if leftover >= 500 else 0
        grew_capital = stock_value + leftover + fd_interest
        semi_annual_return = (grew_capital - capital) / capital if capital else 0.0
        annual_return = (1 + semi_annual_return) ** 2 - 1

        return render_template('optimize_result.html',
                                capital=capital,
                                strategy=strategy,
                                allocation_details=allocation_details,
                                grew_capital=grew_capital,
                                annual_return=annual_return,
                                portfolio_risk=actual_risk,
                                plot_html=plot_html,
                                leftover=leftover,
                                allocation_sum=allocation_sum,
                                fd_rate=MAYBANK_FD_RATE,
                            )
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        return render_template('optimize_form.html', error=str(e))