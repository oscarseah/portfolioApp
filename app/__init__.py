from flask import Flask


def create_app():
    # Create and configure the Flask application

    # Initialize Flask application
    app = Flask(__name__)
    
    # Import and register blueprint
    from .portfolio_bp import bp as portfolio_bp
    app.register_blueprint(portfolio_bp)
    
    # Add number_format filter
    @app.template_filter('number_format')
    def number_format_filter(value, decimals=0):
        # Format numbers with commas and decimals
        try:
            if value is None:
                return ""
            return f"{value:,.{decimals}f}"
        except:
            return value
    
    return app