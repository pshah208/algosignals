"""Earnings factor.

Uses yfinance to get:
- Upcoming earnings date proximity (pre-earnings positive bias)
- EPS surprise from the most recent quarter
- Earnings trend (consensus revision direction)

Degrades gracefully to score=0.0 when data is unavailable.
"""

import datetime

from services.factors import FactorResult
from utils.logging import get_logger

logger = get_logger(__name__)


def compute(symbol: str) -> FactorResult:
    """Compute the earnings score for *symbol*.

    Args:
        symbol: Ticker symbol (yfinance format).

    Returns:
        :class:`~services.factors.FactorResult` with score in [-1, +1].
    """
    result = FactorResult(
        source="yfinance/earnings",
        timestamp=datetime.datetime.utcnow().isoformat(),
    )
    try:
        import numpy as np
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        score = 0.0
        notes: list[str] = []
        raw: dict = {}

        # --- EPS surprise from earnings history ---
        try:
            eh = ticker.earnings_history
            if eh is not None and not (hasattr(eh, "empty") and eh.empty):
                df = eh if hasattr(eh, "iloc") else None
                if df is not None and len(df) > 0:
                    latest = df.iloc[-1]
                    eps_est = latest.get("epsEstimate") or latest.get("EPS Estimate")
                    eps_act = latest.get("epsActual") or latest.get("EPS Actual")
                    if eps_est and eps_act and eps_est != 0:
                        surprise_pct = (float(eps_act) - float(eps_est)) / abs(float(eps_est))
                        raw["eps_surprise_pct"] = round(surprise_pct, 4)
                        s = float(np.clip(surprise_pct * 3, -1, 1))  # 33%+ surprise → ±1
                        score += s * 0.5
                        notes.append(f"eps-surprise={surprise_pct:.1%}({s:+.2f})")
        except Exception:
            logger.exception("earnings_signals: earnings_history failed for %s", symbol)

        # --- Upcoming earnings date ---
        try:
            calendar = ticker.calendar
            cal_dict: dict = {}
            if calendar is not None:
                if hasattr(calendar, "to_dict"):
                    cal_dict = calendar.to_dict()
                elif isinstance(calendar, dict):
                    cal_dict = calendar

            next_earnings = cal_dict.get("Earnings Date") or cal_dict.get("earnings_date")
            if next_earnings:
                try:
                    if isinstance(next_earnings, list):
                        next_earnings = next_earnings[0]
                    ne_dt = (
                        next_earnings.date()
                        if hasattr(next_earnings, "date")
                        else datetime.date.fromisoformat(str(next_earnings)[:10])
                    )
                    days_until = (ne_dt - datetime.date.today()).days
                    raw["days_until_earnings"] = days_until
                    if 0 <= days_until <= 7:
                        score += 0.2
                        notes.append(f"earnings-imminent({days_until}d)")
                    elif 7 < days_until <= 30:
                        score += 0.1
                        notes.append(f"earnings-soon({days_until}d)")
                except Exception:
                    pass
        except Exception:
            logger.exception("earnings_signals: calendar failed for %s", symbol)

        # --- Earnings trend / consensus revisions (from info) ---
        try:
            info = ticker.info or {}
            eps_fwd = info.get("forwardEps")
            eps_trail = info.get("trailingEps")
            if eps_fwd and eps_trail and eps_trail != 0:
                growth = (float(eps_fwd) - float(eps_trail)) / abs(float(eps_trail))
                raw["eps_growth_fwd"] = round(growth, 4)
                s = float(np.clip(growth * 2, -1, 1))
                score += s * 0.3
                notes.append(f"eps-fwd-growth={growth:.1%}({s:+.2f})")
        except Exception:
            logger.exception("earnings_signals: eps growth calc failed for %s", symbol)

        if not notes:
            result.rationale = "Earnings data not available."
            return result

        result.score = round(float(np.clip(score, -1, 1)), 4)
        result.rationale = "; ".join(notes)
        result.raw_values = raw

    except Exception:
        logger.exception("earnings_signals: error processing %s", symbol)
        result.rationale = "data unavailable"

    return result
