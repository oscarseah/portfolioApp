from flask import render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import logging
from .portfolio import get_stock_data, filter_stocks, calculate_efficient_frontier, calculate_steepest_descent, refine_efficient_frontier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize as None to detect initialization failures
all_stocks = None
filtered_in = None
filtered_out = None

def initialize_stock_data():
    """Initialize stock data with proper error handling"""
    global all_stocks, filtered_in, filtered_out
    try:
        all_stocks = get_stock_data() or []
        filtered_in, filtered_out = filter_stocks(all_stocks) if all_stocks else ([], [])
        logger.info(f"Initialized {len(all_stocks)} stocks | {len(filtered_in)} kept | {len(filtered_out)} removed")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        all_stocks, filtered_in, filtered_out = [], [], []
        raise

# Initialize during app startup
initialize_stock_data()

def register_routes(bp):
    """Register all routes with the blueprint"""
    
    @bp.route('/')
    def home():
        return redirect(url_for('portfolio_bp.optimize_portfolio'))

    @bp.route('/optimize', methods=['GET', 'POST'])
    def optimize_portfolio():
        error = None
        if request.method == 'POST':
            try:
                capital = float(request.form.get('capital'))
                strategy = request.form.get('strategy')
                
                # Validate inputs
                if capital < 1000:
                    raise ValueError("Minimum investment is RM 1,000")
                if strategy not in ['conservative', 'balanced', 'aggressive']:
                    raise ValueError("Invalid strategy selected")
                
                # Get filtered stocks
                global filtered_in
                if not filtered_in:
                    initialize_stock_data()
                
                stocks_df = pd.DataFrame(filtered_in)
                
                # Calculate efficient frontier and steepest descent
                raw_frontier = calculate_efficient_frontier(stocks_df)
                frontier_data = refine_efficient_frontier(raw_frontier, stocks_df)
                if not frontier_data:
                    raise ValueError("Could not calculate efficient frontier")
                    
                steepest_port = calculate_steepest_descent(stocks_df)
                if not steepest_port:
                    raise ValueError("Could not calculate minimum risk portfolio")
                steepest_port = steepest_port[0]  # Get first result
                
                # Select portfolio based on strategy
                if strategy == 'conservative':
                    selected_port = next((p for p in frontier_data if p['type'] == 'Low Risk'), steepest_port)
                elif strategy == 'aggressive':
                    selected_port = next((p for p in frontier_data if p['type'] == 'High Risk'), steepest_port)
                else:  # balanced
                    selected_port = next((p for p in frontier_data if p['type'] == 'Medium Risk'), steepest_port)

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
                
                # Save plot to bytes buffer
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
                logger.error(f"Optimization error: {str(e)}")
                error = str(e)
        
        # GET request or error occurred - show form
        return render_template('optimize_form.html', error=error)