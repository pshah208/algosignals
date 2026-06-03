"""Dashboard blueprint — / and /runs routes."""

import json

from flask import Blueprint, flash, redirect, render_template, url_for

from database.db import SessionLocal
from database.models import Recommendation, SignalRun

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
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
                recommendations.append(d)
        return render_template(
            "dashboard.html",
            latest_run=latest_run.to_dict() if latest_run else None,
            recommendations=recommendations,
        )
    finally:
        db.close()


@bp.route("/run-now", methods=["POST"])
def run_now():
    from services.signal_service import run_signals

    try:
        run_id = run_signals()
        flash(f"Signal run #{run_id} completed successfully.", "success")
    except Exception as exc:
        flash(f"Run failed: {exc}", "danger")
    return redirect(url_for("dashboard.index"))


@bp.route("/runs")
def runs():
    db = SessionLocal()
    try:
        all_runs = db.query(SignalRun).order_by(SignalRun.run_at.desc()).limit(50).all()
        return render_template("runs.html", runs=[r.to_dict() for r in all_runs])
    finally:
        db.close()
