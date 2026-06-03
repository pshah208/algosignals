"""Config blueprint — /config routes."""

import json

from flask import Blueprint, flash, redirect, render_template, request, url_for

from database.db import SessionLocal
from database.models import AppConfig
from services.signal_service import DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD, DEFAULT_WEIGHTS

bp = Blueprint("config_routes", __name__, url_prefix="/config")


def _get(db, key: str, default: str = "") -> str:
    row = db.query(AppConfig).filter_by(key=key).first()
    return row.value if row else default


def _set(db, key: str, value: str) -> None:
    row = db.query(AppConfig).filter_by(key=key).first()
    if row:
        row.value = value
    else:
        db.add(AppConfig(key=key, value=value))


@bp.route("/")
def index():
    db = SessionLocal()
    try:
        raw_weights = _get(db, "factor_weights", json.dumps(DEFAULT_WEIGHTS))
        try:
            weights = json.loads(raw_weights)
        except ValueError:
            weights = dict(DEFAULT_WEIGHTS)

        return render_template(
            "config.html",
            weights=weights,
            buy_threshold=float(_get(db, "buy_threshold", str(DEFAULT_BUY_THRESHOLD))),
            sell_threshold=float(_get(db, "sell_threshold", str(DEFAULT_SELL_THRESHOLD))),
            schedule_hour=int(_get(db, "schedule_hour_ist", "9")),
            schedule_minute=int(_get(db, "schedule_minute_ist", "0")),
            scheduler_enabled=_get(db, "scheduler_enabled", "true") == "true",
        )
    finally:
        db.close()


@bp.route("/save", methods=["POST"])
def save():
    db = SessionLocal()
    try:
        # Weights
        weights = {}
        for factor in DEFAULT_WEIGHTS:
            val = request.form.get(f"weight_{factor}", "")
            try:
                weights[factor] = round(float(val), 4)
            except ValueError:
                weights[factor] = DEFAULT_WEIGHTS[factor]
        _set(db, "factor_weights", json.dumps(weights))

        # Thresholds
        try:
            buy_thr = float(request.form.get("buy_threshold", str(DEFAULT_BUY_THRESHOLD)))
        except ValueError:
            buy_thr = DEFAULT_BUY_THRESHOLD
        try:
            sell_thr = float(request.form.get("sell_threshold", str(DEFAULT_SELL_THRESHOLD)))
        except ValueError:
            sell_thr = DEFAULT_SELL_THRESHOLD
        _set(db, "buy_threshold", str(buy_thr))
        _set(db, "sell_threshold", str(sell_thr))

        # Schedule
        try:
            hour = int(request.form.get("schedule_hour", "9"))
            minute = int(request.form.get("schedule_minute", "0"))
        except ValueError:
            hour, minute = 9, 0
        _set(db, "schedule_hour_ist", str(hour))
        _set(db, "schedule_minute_ist", str(minute))
        scheduler_enabled = request.form.get("scheduler_enabled") == "on"
        _set(db, "scheduler_enabled", "true" if scheduler_enabled else "false")

        db.commit()

        # Re-schedule if enabled
        if scheduler_enabled:
            from services.scheduler import schedule_job

            schedule_job(hour, minute)
            flash(f"Scheduler updated to IST {hour:02d}:{minute:02d}.", "info")
        else:
            from services.scheduler import get_scheduler

            sched = get_scheduler()
            if sched.get_job("daily_signals"):
                sched.remove_job("daily_signals")
                flash("Scheduler disabled.", "info")

        flash("Configuration saved.", "success")
    except Exception as exc:
        flash(f"Error saving config: {exc}", "danger")
    finally:
        db.close()
    return redirect(url_for("config_routes.index"))
