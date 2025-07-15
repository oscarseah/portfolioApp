from flask import render_template, Blueprint
from app.portfolio import get_stock_data, filter_stocks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('portfolio_bp', __name__)

# Initialize empty stock lists
all_stocks = []
filtered_in = []
filtered_out = []

try:
    # Load and filter stocks once at startup
    all_stocks = get_stock_data()
    logger.info(f"Loaded {len(all_stocks)} stocks")
    
    if all_stocks:
        filtered_in, filtered_out = filter_stocks(all_stocks)
        logger.info(f"Filtered to {len(filtered_in)} kept and {len(filtered_out)} removed")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    all_stocks = []
    filtered_in = []
    filtered_out = []

@bp.route('/')
def index():
    # Render template with filtering results
    return render_template('index.html',
                         all_stocks=all_stocks,
                         filtered_in=filtered_in,
                         filtered_out=filtered_out)