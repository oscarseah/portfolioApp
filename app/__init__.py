from flask import Flask
from app.routes import bp

def create_app():
    # Initialize Flask application
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(bp)
    
    return app