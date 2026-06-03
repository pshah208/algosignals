"""Corporate events / actions factor.

Uses yfinance to fetch:
- Upcoming / recent dividends
- Stock splits
- Calendar events (earnings, dividends from calendar)

Positive signals: dividend increases, upcoming earnings with upward revision, index inclusion.
Negative signals: dividend cuts, reverse splits.

Degrades gracefully to score=0.0 when data is unavailable.
"""

import datetime

from services.factors import FactorResult
from utils.logging import get_logger

logger = get_logger(__name__)


def compute(symbol: str) -> FactorResult:
    """Compute the corporate-events score for *symbol*.

    Args:
        symbol: Ticker symbol (yfinance format).

    Returns:
        :class:`~services.factors.FactorResult` with score in [-1, +1].
    """
    result = FactorResult(
        source="yfinance/events",
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    try:
        import numpy as np
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        score = 0.0
        notes: list[str] = []
        raw: dict = {}

        # --- Dividends ---
        try:
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                recent = dividends.last("180D")
                if not recent.empty:
                    raw["recent_dividends"] = float(recent.iloc[-1])
                    score += 0.2  # paying dividends is positive
                    notes.append("dividend-paying")
                    # Check if most recent dividend is higher than 1y ago
                    hist_div = dividends.last("365D")
                    if len(hist_div) >= 2:
                        if hist_div.iloc[-1] > hist_div.iloc[-2]:
                            score += 0.1
                            notes.append("dividend-growing")
                        elif hist_div.iloc[-1] < hist_div.iloc[-2]:
                            score -= 0.2
                            notes.append("dividend-cut")
        except Exception:
            logger.exception("events_signals: dividends fetch failed for %s", symbol)

        # --- Splits ---
        try:
            splits = ticker.splits
            if splits is not None and not splits.empty:
                recent_splits = splits.last("180D")
                if not recent_splits.empty:
                    ratio = float(recent_splits.iloc[-1])
                    raw["recent_split_ratio"] = ratio
                    if ratio > 1:
                        score += 0.15  # forward split → bullish signal
                        notes.append(f"forward-split({ratio:.0f}:1)")
                    elif ratio < 1:
                        score -= 0.25  # reverse split → bearish
                        notes.append(f"reverse-split(1:{1/ratio:.0f})")
        except Exception:
            logger.exception("events_signals: splits fetch failed for %s", symbol)

        # --- Calendar (earnings / ex-div dates) ---
        try:
            calendar = ticker.calendar
            if calendar is not None and not isinstance(calendar, type(None)):
                # yfinance returns a dict-like object; handle both old/new API shapes
                cal_dict = {}
                if hasattr(calendar, "to_dict"):
                    cal_dict = calendar.to_dict()
                elif isinstance(calendar, dict):
                    cal_dict = calendar

                next_earnings = cal_dict.get("Earnings Date") or cal_dict.get("earnings_date")
                if next_earnings:
                    raw["next_earnings"] = str(next_earnings)
                    # Upcoming earnings within 14 days → slight positive (attention)
                    try:
                        if isinstance(next_earnings, list):
                            next_earnings = next_earnings[0]
                        ne_dt = (
                            next_earnings.date()
                            if hasattr(next_earnings, "date")
                            else datetime.date.fromisoformat(str(next_earnings)[:10])
                        )
                        days_until = (ne_dt - datetime.date.today()).days
                        if 0 <= days_until <= 14:
                            score += 0.1
                            notes.append(f"earnings-in-{days_until}d")
                    except Exception:
                        pass
        except Exception:
            logger.exception("events_signals: calendar fetch failed for %s", symbol)

        result.score = round(float(np.clip(score, -1, 1)), 4)
        result.rationale = ", ".join(notes) if notes else "No significant corporate events."
        result.raw_values = raw

    except Exception:
        logger.exception("events_signals: error processing %s", symbol)
        result.rationale = "data unavailable"

    return result
