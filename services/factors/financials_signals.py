"""Fundamentals / financials factor.

Fetches basic fundamentals via yfinance:
- Trailing P/E (low → value, high → expensive)
- Revenue growth YoY
- Profit margin
- Debt-to-equity ratio

Each metric is mapped to a [-1, +1] contribution and averaged.
Degrades gracefully to score=0.0 when data is unavailable.
"""

import datetime

from services.factors import FactorResult
from utils.logging import get_logger

logger = get_logger(__name__)


def _norm_pe(pe: float) -> float:
    """Map trailing P/E to a score.

    Below 15 → positive (value), above 40 → negative (expensive),
    between → linear interpolation.
    """
    if pe <= 0:
        return -0.1  # negative earnings
    if pe < 15:
        return 0.5
    if pe > 40:
        return -0.5
    # linear: 15→+0.5, 40→-0.5
    return round(0.5 - (pe - 15) / 25, 4)


def _norm_margin(margin: float) -> float:
    """Map profit margin (0..1) to a score."""
    if margin < 0:
        return max(-1.0, margin * 2)
    return round(min(1.0, margin * 3), 4)  # 33%+ margin → +1


def _norm_growth(growth: float) -> float:
    """Map revenue growth rate to a score."""
    return round(max(-1.0, min(1.0, growth * 4)), 4)  # 25%+ growth → +1


def _norm_debt_equity(de: float) -> float:
    """Map D/E ratio to a score (lower → better)."""
    if de < 0:
        return -0.3
    if de < 0.5:
        return 0.4
    if de < 1.5:
        return 0.1
    if de < 3.0:
        return -0.2
    return -0.5


def compute(symbol: str) -> FactorResult:
    """Compute the fundamentals score for *symbol*.

    Args:
        symbol: Ticker symbol (yfinance format).

    Returns:
        :class:`~services.factors.FactorResult` with score in [-1, +1].
    """
    result = FactorResult(
        source="yfinance/fundamentals",
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    try:
        import numpy as np
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        if not info:
            result.rationale = "No fundamental data available."
            return result

        scores: list[float] = []
        notes: list[str] = []
        raw: dict = {}

        # P/E
        pe = info.get("trailingPE")
        if pe is not None:
            s = _norm_pe(float(pe))
            scores.append(s)
            notes.append(f"PE={pe:.1f}({s:+.2f})")
            raw["trailing_pe"] = pe

        # Revenue growth
        rev_growth = info.get("revenueGrowth")
        if rev_growth is not None:
            s = _norm_growth(float(rev_growth))
            scores.append(s)
            notes.append(f"rev-growth={rev_growth:.1%}({s:+.2f})")
            raw["revenue_growth"] = rev_growth

        # Profit margin
        margin = info.get("profitMargins")
        if margin is not None:
            s = _norm_margin(float(margin))
            scores.append(s)
            notes.append(f"margin={margin:.1%}({s:+.2f})")
            raw["profit_margin"] = margin

        # Debt/equity
        de = info.get("debtToEquity")
        if de is not None:
            s = _norm_debt_equity(float(de) / 100)  # yfinance reports as percentage
            scores.append(s)
            notes.append(f"D/E={de:.1f}({s:+.2f})")
            raw["debt_to_equity"] = de

        if not scores:
            result.rationale = "Fundamental metrics not available."
            return result

        result.score = round(float(np.clip(sum(scores) / len(scores), -1, 1)), 4)
        result.rationale = "; ".join(notes)
        result.raw_values = raw

    except Exception:
        logger.exception("financials_signals: error processing %s", symbol)
        result.rationale = "data unavailable"

    return result
