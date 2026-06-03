"""Signal service: orchestrates the five factors → composite score → recommendation.

Usage::

    from services.signal_service import run_signals
    run_id = run_signals()
"""

import datetime
import json
from typing import Any

from database.db import SessionLocal
from database.models import AppConfig, Recommendation, SignalRun, Watchlist
from services.factors import FactorResult
from services.llm.llm_client import get_llm_client
from utils.logging import get_logger

logger = get_logger(__name__)

# Default factor weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "technical": 0.35,
    "news": 0.25,
    "events": 0.10,
    "financials": 0.20,
    "earnings": 0.10,
}

# Default score thresholds
DEFAULT_BUY_THRESHOLD = 0.15
DEFAULT_SELL_THRESHOLD = -0.15


def _get_config(db, key: str, default: Any) -> Any:
    """Read a value from the AppConfig table."""
    row = db.query(AppConfig).filter_by(key=key).first()
    return row.value if row else default


def _get_weights(db) -> dict[str, float]:
    """Load factor weights from DB, falling back to defaults."""
    raw = _get_config(db, "factor_weights", None)
    if raw:
        try:
            w = json.loads(raw)
            # Ensure all keys present
            for k, v in DEFAULT_WEIGHTS.items():
                w.setdefault(k, v)
            return w
        except (ValueError, TypeError):
            pass
    return dict(DEFAULT_WEIGHTS)


def _score_to_action(score: float, buy_thr: float, sell_thr: float) -> str:
    if score >= buy_thr:
        return "BUY"
    if score <= sell_thr:
        return "SELL"
    return "HOLD"


def _run_factor(name: str, symbol: str, llm_client: Any) -> FactorResult:
    """Run a single factor module, catching all exceptions."""
    try:
        if name == "technical":
            from services.factors import technical_signals

            return technical_signals.compute(symbol)
        if name == "news":
            from services.factors import news_signals

            return news_signals.compute(symbol, llm_client=llm_client)
        if name == "events":
            from services.factors import events_signals

            return events_signals.compute(symbol)
        if name == "financials":
            from services.factors import financials_signals

            return financials_signals.compute(symbol)
        if name == "earnings":
            from services.factors import earnings_signals

            return earnings_signals.compute(symbol)
    except Exception:
        logger.exception("Factor %s failed for %s", name, symbol)
    return FactorResult(score=0.0, rationale="data unavailable")


def run_signals() -> int:
    """Execute one full signal run for all enabled watchlist symbols.

    Returns:
        The ``id`` of the newly created :class:`~database.models.SignalRun` row.
    """
    db = SessionLocal()
    run = SignalRun(run_at=datetime.datetime.utcnow(), status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    try:
        symbols = (
            db.query(Watchlist).filter_by(enabled=True).all()
        )
        if not symbols:
            run.status = "ok"
            run.summary = "No enabled symbols in watchlist."
            db.commit()
            return run_id

        weights = _get_weights(db)
        buy_thr = float(_get_config(db, "buy_threshold", DEFAULT_BUY_THRESHOLD))
        sell_thr = float(_get_config(db, "sell_threshold", DEFAULT_SELL_THRESHOLD))

        llm_client = get_llm_client()
        results_summary: list[str] = []

        for item in symbols:
            symbol = item.symbol
            logger.info("Processing %s …", symbol)

            factor_scores: dict[str, float] = {}
            factor_notes: dict[str, str] = {}
            factor_details: dict[str, dict] = {}

            for factor_name in DEFAULT_WEIGHTS:
                fr: FactorResult = _run_factor(factor_name, symbol, llm_client)
                factor_scores[factor_name] = round(fr.score, 4)
                factor_notes[factor_name] = fr.rationale
                factor_details[factor_name] = {
                    "score": fr.score,
                    "rationale": fr.rationale,
                    "source": fr.source,
                    "raw_values": fr.raw_values,
                }

            # Weighted composite
            composite = sum(
                weights.get(k, DEFAULT_WEIGHTS[k]) * v for k, v in factor_scores.items()
            )
            composite = round(max(-1.0, min(1.0, composite)), 4)
            action = _score_to_action(composite, buy_thr, sell_thr)

            # LLM rationale
            try:
                rationale = llm_client.generate_rationale(
                    symbol=symbol,
                    factor_scores=factor_scores,
                    factor_notes=factor_notes,
                    composite=composite,
                    action=action,
                )
            except Exception:
                logger.exception("LLM rationale generation failed for %s", symbol)
                rationale = llm_client._fallback_rationale(
                    symbol, factor_scores, composite, action
                )

            rec = Recommendation(
                run_id=run_id,
                symbol=symbol,
                exchange=item.exchange,
                action=action,
                composite_score=composite,
                factor_scores=json.dumps(factor_details),
                rationale=rationale,
                llm_used=llm_client.enabled,
                created_at=datetime.datetime.utcnow(),
            )
            db.add(rec)
            results_summary.append(f"{symbol}:{action}({composite:+.2f})")
            logger.info("%s → %s (composite=%.4f)", symbol, action, composite)

        db.commit()
        run.status = "ok"
        run.summary = ", ".join(results_summary)
        db.commit()
        logger.info("Signal run %d completed: %s", run_id, run.summary)

    except Exception:
        logger.exception("Signal run %d failed", run_id)
        try:
            run.status = "error"
            run.summary = "Run failed — see logs."
            db.commit()
        except Exception:
            pass
    finally:
        db.close()

    return run_id
