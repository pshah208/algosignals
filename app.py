"""AlgoSignals — Flask application factory.

Run with::

    python app.py

Or via gunicorn (single worker recommended for APScheduler)::

    gunicorn -w 1 app:app
"""

import atexit

from flask import Flask

from config import settings
from database.db import init_db
from utils.logging import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured :class:`flask.Flask` instance.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = settings.SECRET_KEY

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    with app.app_context():
        init_db()

    # ------------------------------------------------------------------
    # Blueprints
    # ------------------------------------------------------------------
    from blueprints.dashboard import bp as dashboard_bp
    from blueprints.watchlist import bp as watchlist_bp
    from blueprints.config_routes import bp as config_bp
    from blueprints.api import bp as api_bp
    from blueprints.models import bp as models_bp
    from blueprints.predict import bp as predict_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(watchlist_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(predict_bp)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    from services.scheduler import init_scheduler, shutdown_scheduler

    # Read persisted schedule time (fall back to env defaults)
    from database.db import SessionLocal
    from database.models import AppConfig

    db = SessionLocal()
    try:
        def _cfg(key: str, default: str) -> str:
            row = db.query(AppConfig).filter_by(key=key).first()
            return row.value if row else default

        hour_ist = int(_cfg("schedule_hour_ist", str(settings.SCHEDULE_HOUR_IST)))
        minute_ist = int(_cfg("schedule_minute_ist", str(settings.SCHEDULE_MINUTE_IST)))
        scheduler_enabled = _cfg("scheduler_enabled", "true") == "true"
    finally:
        db.close()

    if scheduler_enabled:
        init_scheduler(hour_ist, minute_ist)
        atexit.register(shutdown_scheduler)

    logger.info(
        "AlgoSignals started. LLM enabled=%s. Scheduler enabled=%s.",
        settings.llm_enabled,
        scheduler_enabled,
    )
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=settings.FLASK_ENV == "development", use_reloader=False, port=5000)
