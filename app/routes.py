from flask import render_template, Blueprint
import logging
from .portfolio import get_stock_data, filter_stocks  # Changed to relative import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('portfolio_bp', __name__)

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
        raise  # Re-raise if you want startup to fail

# Initialize during app startup
initialize_stock_data()

@bp.route('/')
def index():
    # Ensure data exists before rendering
    if all_stocks is None:
        initialize_stock_data()
        
    return render_template('index.html',
                         all_stocks=all_stocks or [],
                         filtered_in=filtered_in or [],
                         filtered_out=filtered_out or [])