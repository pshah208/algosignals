"""Read-only JSON API blueprint — /api/v1/recommendations.

Protected by a simple app API key passed as a ****** or
``?api_key=`` query parameter.

This endpoint is intentionally READ-ONLY and is NOT wired to any
order placement system.
"""

from functools import wraps

from flask import Blueprint, jsonify, request

from config import settings
from database.db import SessionLocal
from database.models import Recommendation, SignalRun

bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not settings.APP_API_KEY:
            # If no key configured, API is open (dev convenience)
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        bearer = auth_header.removeprefix("Bearer ").strip()
        query_key = request.args.get("api_key", "")
        if bearer != settings.APP_API_KEY and query_key != settings.APP_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated


@bp.route("/recommendations")
@_require_api_key
def recommendations():
    """Return the latest run's recommendations as JSON.

    This is a **read-only** endpoint. No orders are placed.
    """
    db = SessionLocal()
    try:
        latest_run = db.query(SignalRun).order_by(SignalRun.run_at.desc()).first()
        if not latest_run:
            return jsonify({"run": None, "recommendations": [], "disclaimer": _disclaimer()})

        recs = (
            db.query(Recommendation)
            .filter_by(run_id=latest_run.id)
            .order_by(Recommendation.composite_score.desc())
            .all()
        )
        return jsonify(
            {
                "run": latest_run.to_dict(),
                "recommendations": [r.to_dict() for r in recs],
                "disclaimer": _disclaimer(),
            }
        )
    finally:
        db.close()


def _disclaimer() -> str:
    return (
        "AlgoSignals is a research tool only. "
        "This is NOT investment advice and does NOT place brokerage orders."
    )
