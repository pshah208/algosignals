"""Signal service: orchestrates the five factors → composite score → recommendation.

Usage::

    from services.signal_service import run_signals
    run_id = run_signals()
"""

import datetime
import json
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import Lock
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

MAX_SYMBOL_WORKERS = 4
PRICE_CACHE_TTL_SECONDS = 60
PRICE_CACHE_MAXSIZE = 256
_price_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_price_cache_lock = Lock()


def _clear_price_cache() -> None:
    with _price_cache_lock:
        _price_cache.clear()


def get_current_price(symbol: str) -> dict[str, Any]:
    """Fetch prices with a bounded, monotonic TTL cache; do not cache failures."""
    with _price_cache_lock:
        cached = _price_cache.get(symbol)
        if cached is not None:
            expires_at, result = cached
            if time.monotonic() < expires_at:
                _price_cache.move_to_end(symbol)
                return dict(result)
            del _price_cache[symbol]

    result = _fetch_current_price(symbol)
    if result["price"] is not None:
        with _price_cache_lock:
            _price_cache[symbol] = (
                time.monotonic() + PRICE_CACHE_TTL_SECONDS, dict(result)
            )
            _price_cache.move_to_end(symbol)
            while len(_price_cache) > PRICE_CACHE_MAXSIZE:
                _price_cache.popitem(last=False)
    return result


get_current_price.cache_clear = _clear_price_cache


def _fetch_current_price(symbol: str) -> dict[str, Any]:
    """Best-effort latest-price fetch for dashboard display."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        fast = getattr(ticker, "fast_info", None) or {}

        price = (
            fast.get("last_price")
            or fast.get("lastPrice")
            or fast.get("regularMarketPrice")
            or fast.get("previous_close")
            or fast.get("last_close")
        )
        currency = fast.get("currency") or ""

        if price is None:
            info = getattr(ticker, "info", None) or {}
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )
            currency = currency or (info.get("currency") or "")

        if price is None:
            return {"price": None, "currency": currency}
        return {"price": round(float(price), 4), "currency": currency}
    except Exception:
        logger.debug("Price lookup failed for %s", symbol, exc_info=True)
        return {"price": None, "currency": ""}


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


def _compute_symbol(
    symbol: str,
    exchange: str,
    weights: dict[str, float],
    buy_thr: float,
    sell_thr: float,
    llm_client: Any,
) -> dict[str, Any]:
    """Compute one symbol sequentially without accessing ORM objects."""
    logger.info("Processing %s …", symbol)
    factor_scores: dict[str, float] = {}
    factor_notes: dict[str, str] = {}
    factor_details: dict[str, dict] = {}
    for factor_name in DEFAULT_WEIGHTS:
        fr = _run_factor(factor_name, symbol, llm_client)
        factor_scores[factor_name] = round(fr.score, 4)
        factor_notes[factor_name] = fr.rationale
        factor_details[factor_name] = {
            "score": fr.score,
            "rationale": fr.rationale,
            "source": fr.source,
            "raw_values": fr.raw_values,
        }

    composite = sum(
        weights.get(k, DEFAULT_WEIGHTS[k]) * v for k, v in factor_scores.items()
    )
    composite = round(max(-1.0, min(1.0, composite)), 4)
    action = _score_to_action(composite, buy_thr, sell_thr)
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
    logger.info("%s → %s (composite=%.4f)", symbol, action, composite)
    return {
        "symbol": symbol,
        "exchange": exchange,
        "action": action,
        "composite_score": composite,
        "factor_scores": json.dumps(factor_details),
        "rationale": rationale,
        "llm_used": llm_client.enabled,
        "created_at": datetime.datetime.utcnow(),
    }


def _compute_symbols(symbols, weights, buy_thr, sell_thr, llm_client):
    """Keep at most four tasks in flight and yield in watchlist order."""
    symbols = iter(symbols)
    with ThreadPoolExecutor(max_workers=MAX_SYMBOL_WORKERS) as executor:
        def submit(item):
            return executor.submit(
                _compute_symbol, *item, weights, buy_thr, sell_thr, llm_client
            )

        pending = deque(submit(item) for item in islice(symbols, MAX_SYMBOL_WORKERS))
        while pending:
            yield pending.popleft().result()
            item = next(symbols, None)
            if item is not None:
                pending.append(submit(item))


def run_signals(llm_client: Any = None) -> int:
    """Execute one full signal run for all enabled watchlist symbols.

    Returns:
        The ``id`` of the newly created :class:`~database.models.SignalRun` row.

    An injected LLM client is used only for this run, without changing the default.
    """
    db = SessionLocal()
    run = SignalRun(run_at=datetime.datetime.utcnow(), status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    try:
        symbols = [
            (item.symbol, item.exchange)
            for item in db.query(Watchlist).filter_by(enabled=True).all()
        ]
        if not symbols:
            run.status = "ok"
            run.summary = "No enabled symbols in watchlist."
            db.commit()
            return run_id

        weights = _get_weights(db)
        buy_thr = float(_get_config(db, "buy_threshold", DEFAULT_BUY_THRESHOLD))
        sell_thr = float(_get_config(db, "sell_threshold", DEFAULT_SELL_THRESHOLD))

        if llm_client is None:
            llm_client = get_llm_client()
        results_summary: list[str] = []

        for result in _compute_symbols(
            symbols, weights, buy_thr, sell_thr, llm_client
        ):
            db.add(Recommendation(run_id=run_id, **result))
            results_summary.append(
                f"{result['symbol']}:{result['action']}"
                f"({result['composite_score']:+.2f})"
            )

        db.commit()
        run.status = "ok"
        run.summary = ", ".join(results_summary)
        db.commit()
        logger.info("Signal run %d completed: %s", run_id, run.summary)

    except Exception:
        logger.exception("Signal run %d failed", run_id)
        try:
            db.rollback()
            run.status = "error"
            run.summary = "Run failed — see logs."
            db.commit()
        except Exception:
            pass
    finally:
        db.close()

    return run_id
