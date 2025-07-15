from flask import render_template, request, Blueprint
import numpy as np
from app.portfolio import calculate_efficient_frontier, calculate_steepest_descent, get_stock_data
import base64
import matplotlib
matplotlib.use('Agg') # Configure matplotlib to use non-GUI backend (required for server environments)
import matplotlib.pyplot as plt
import io

bp = Blueprint('portfolio_bp', __name__)

@bp.route('/', methods=['GET', 'POST'])
def index():
    stocks = get_stock_data()

    # Collect form data with default values
    form_data = {
        'num_stocks': request.form.get('num_stocks', ''),
        'selected_stocks': request.form.getlist('stocks'),
        'capital': request.form.get('capital', ''),
        'optimization_method': request.form.get('optimization_method', 'efficient_frontier')

    }

    # Initialize result variables
    results = None
    plot_url = None
    error = None

    # Handle form submission
    if request.method == 'POST':
        try:
            # Convert and validate form inputs
            capital = float(form_data['capital']) # Convert to monetary value
            num_stocks = int(form_data['num_stocks']) # Convert to integer
            selected_stocks = [s for s in form_data['selected_stocks'] if s] # Filter empty values
            
            # Validate stock selection count matches input
            if len(selected_stocks) != num_stocks:
                raise ValueError(f"Select exactly {num_stocks} stocks")

             # Calculate portfolio suggestions using Modern Portfolio Theory
            method = form_data['optimization_method']
            if method == 'efficient_frontier':
                suggestions = calculate_efficient_frontier(selected_stocks)
            elif method == 'steepest_descent':
                suggestions = calculate_steepest_descent(selected_stocks)
            else:
                raise ValueError("Invalid optimization method")
            # Generate plot
            fig, ax = plt.subplots()
            
            # Plot stocks
            stock_info = {s['symbol']: s for s in stocks}
            stock_risks = [stock_info[sym]['risk'] for sym in selected_stocks]
            stock_returns = [stock_info[sym]['return'] for sym in selected_stocks]  # Fixed column
            
            ax.scatter(stock_risks, stock_returns, c='gray', label='Stocks')

            if method == 'efficient_frontier':
                # Original plotting for 3 portfolios
                portfolio_risks = [p['risk'] for p in suggestions]
                portfolio_returns = [p['return'] for p in suggestions]
                ax.scatter(portfolio_risks, portfolio_returns, c=['green', 'blue', 'red'], s=150, marker='D', label=['Conservative', 'Balanced', 'Aggressive'])
            else:  # steepest_descent
                # Plot single optimized portfolio
                portfolio = suggestions[0]
                ax.scatter(portfolio['risk'], portfolio['return'], c='purple', s=200, marker='*', label='Optimized Portfolio')
            
            handles, labels = ax.get_legend_handles_labels()
            if method == 'steepest_descent':
                ax.legend(handles[:-1] + [handles[-1]], labels[:-1] + ['Steepest Descent'])

            # Configure plot aesthetics
            ax.set_title('Portfolio Optimization')
            ax.set_xlabel('Risk')
            ax.set_ylabel('Return')
            ax.legend()

            # Save plot
            img = io.BytesIO()
            fig.savefig(img, format='png')
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

            # Format results
            results = [{
                'return_pct': s['return'] * 100,  # Percentage value
                'risk_pct': s['risk'] * 100,      # Percentage value
                'return_amt': capital * s['return'],  # Monetary value
                'risk_amt': capital * s['risk'],      # Monetary value
                'allocation': s['allocation']
            } for s in suggestions]

        # Error handling
        except Exception as e:
            error = str(e)

    # Render template with all context variables
    return render_template('index.html',
                         stocks=stocks,
                         results=results,
                         plot_url=plot_url,
                         error=error,
                         **form_data)