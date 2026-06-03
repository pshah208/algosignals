"""Watchlist blueprint — /watchlist routes."""

from flask import Blueprint, flash, redirect, render_template, request, url_for

from database.db import SessionLocal
from database.models import Watchlist

bp = Blueprint("watchlist", __name__, url_prefix="/watchlist")


@bp.route("/")
def index():
    db = SessionLocal()
    try:
        items = db.query(Watchlist).order_by(Watchlist.symbol).all()
        return render_template("watchlist.html", items=[i.to_dict() for i in items])
    finally:
        db.close()


@bp.route("/add", methods=["POST"])
def add():
    symbol = request.form.get("symbol", "").strip().upper()
    exchange = request.form.get("exchange", "").strip().upper()
    if not symbol:
        flash("Symbol is required.", "warning")
        return redirect(url_for("watchlist.index"))

    db = SessionLocal()
    try:
        existing = db.query(Watchlist).filter_by(symbol=symbol).first()
        if existing:
            existing.enabled = True
            existing.exchange = exchange or existing.exchange
            flash(f"{symbol} already exists; re-enabled.", "info")
        else:
            db.add(Watchlist(symbol=symbol, exchange=exchange, enabled=True))
            flash(f"{symbol} added to watchlist.", "success")
        db.commit()
    finally:
        db.close()
    return redirect(url_for("watchlist.index"))


@bp.route("/toggle/<int:item_id>", methods=["POST"])
def toggle(item_id: int):
    db = SessionLocal()
    try:
        item = db.query(Watchlist).filter_by(id=item_id).first()
        if item:
            item.enabled = not item.enabled
            db.commit()
            flash(f"{item.symbol} {'enabled' if item.enabled else 'disabled'}.", "info")
        else:
            flash("Symbol not found.", "warning")
    finally:
        db.close()
    return redirect(url_for("watchlist.index"))


@bp.route("/delete/<int:item_id>", methods=["POST"])
def delete(item_id: int):
    db = SessionLocal()
    try:
        item = db.query(Watchlist).filter_by(id=item_id).first()
        if item:
            symbol = item.symbol
            db.delete(item)
            db.commit()
            flash(f"{symbol} removed from watchlist.", "success")
        else:
            flash("Symbol not found.", "warning")
    finally:
        db.close()
    return redirect(url_for("watchlist.index"))
