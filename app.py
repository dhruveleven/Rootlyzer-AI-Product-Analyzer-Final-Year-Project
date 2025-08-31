import os
from pathlib import Path
from flask import Flask

from routes.main_routes import main_bp
from routes.report_routes import report_bp
from routes.dashboard_routes import dashboard_bp
from routes.anomaly_routes import anomaly_bp
from routes.rootcause_routes import rootcause_bp
from routes.forecast_routes import forecast_bp
from routes.benchmark_routes import benchmark_bp
from routes.recommendation_routes import recommendation_bp
from routes.demo_routes import demo_bp

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(anomaly_bp)
    app.register_blueprint(rootcause_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(benchmark_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(demo_bp)
    #print(app.url_map)


    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
