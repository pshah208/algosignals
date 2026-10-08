"""Dashboard blueprint — / and /runs routes."""

import json

from flask import Blueprint, flash, redirect, render_template, url_for

from config import settings
from database.db import SessionLocal
from database.models import Recommendation, SignalRun
from services.signal_service import (
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    DEFAULT_WEIGHTS,
    get_current_price,
)

bp = Blueprint("dashboard", __name__)


def _github_context():
    from services.github_auth import get_login_info

    return {
        "github_login": get_login_info(),
        "auth_enabled": bool(
            settings.GITHUB_CLIENT_ID and settings.GITHUB_CLIENT_SECRET and settings.GITHUB_REDIRECT_URI
        ),
    }


@bp.route("/")
def index():
    from blueprints.models import available_models, get_active_model

    db = SessionLocal()
    try:
        latest_run = db.query(SignalRun).order_by(SignalRun.run_at.desc()).first()
        recommendations = []
        if latest_run:
            recs = (
                db.query(Recommendation)
                .filter_by(run_id=latest_run.id)
                .order_by(Recommendation.composite_score.desc())
                .all()
            )
            for r in recs:
                d = r.to_dict()
                d["factor_scores"] = r.factor_scores_dict
                px = get_current_price(r.symbol)
                d["current_price"] = px.get("price")
                d["currency"] = px.get("currency", "")
                recommendations.append(d)
        return render_template(
            "dashboard.html",
            latest_run=latest_run.to_dict() if latest_run else None,
            recommendations=recommendations,
            available_models=available_models(),
            active_model=get_active_model(),
            llm_enabled=settings.llm_enabled,
            llm_provider=settings.LLM_PROVIDER,
            **_github_context(),
        )
    finally:
        db.close()


@bp.route("/metrics")
def metrics():
    """User-facing explanation of factor scoring and composite thresholds."""
    return render_template(
        "metrics.html",
        weights=DEFAULT_WEIGHTS,
        buy_threshold=DEFAULT_BUY_THRESHOLD,
        sell_threshold=DEFAULT_SELL_THRESHOLD,
    )


@bp.route("/run-now", methods=["POST"])
def run_now():
    from blueprints.models import get_active_model
    from services.llm.llm_client import get_llm_client
    from services.signal_service import run_signals
    if settings.LLM_PROVIDER == "copilot":
        from blueprints.research import require_copilot_key

        denied = require_copilot_key(form=True)
        if denied:
            return denied

    try:
        run_id = run_signals(llm_client=get_llm_client(model=get_active_model()))
        flash(f"Signal run #{run_id} completed successfully.", "success")
    except Exception as exc:
        flash(f"Run failed: {exc}", "danger")
    return redirect(url_for("dashboard.index"))


@bp.route("/research")
def research_page():
    return render_template("research.html", llm_provider=settings.LLM_PROVIDER, **_github_context())


@bp.route("/runs")
def runs():
    db = SessionLocal()
    try:
        all_runs = db.query(SignalRun).order_by(SignalRun.run_at.desc()).limit(50).all()
        return render_template("runs.html", runs=[r.to_dict() for r in all_runs])
    finally:
        db.close()
