from flask import Flask
from config.settings import settings
from core.tracing import tracing_manager
from routes.web import web_bp
from routes.api import api_bp
import structlog
import os

# Configure logging
logger = structlog.get_logger()

def create_app():
    """Application factory"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = settings.DEBUG
    
    # Register blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return "Page not found", 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error", error=str(error))
        return "Internal server error", 500
    
    # Health check
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'docker-agent'}
    
    logger.info("Flask application created successfully")
    return app

if __name__ == '__main__':
    app = create_app()
    
    logger.info(
        "Starting Docker Agent application",
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )
    
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )