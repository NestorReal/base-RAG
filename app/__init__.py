import os # Add this import
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    # Construct the absolute path to the templates folder
    # This assumes 'app' directory is inside 'flask_app', and 'templates' is next to 'app'
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    
    # Pass the template_folder argument to the Flask constructor
    app = Flask(__name__, template_folder=template_dir)
    
    CORS(app)
    app.config.from_object('app.config.Config')

    db.init_app(app)

    with app.app_context():
        from .routes import main_routes, document_routes
        app.register_blueprint(main_routes.bp)
        app.register_blueprint(document_routes.bp)
        db.create_all()

    return app