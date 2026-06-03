"""Technical / price factor.

Fetches OHLCV history via yfinance and computes:
- MA crossover (20-day vs 50-day SMA)
- RSI(14)
- Price momentum (20-day return)
- ATR-based normalised volatility penalty

Each sub-signal is normalised to [-1, +1] and blended into a composite score.
Degrades gracefully to score=0.0 when data is unavailable.
"""

import datetime

import numpy as np
import pandas as pd

from services.factors import FactorResult
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure-pandas indicator helpers
# ---------------------------------------------------------------------------


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Main factor function
# ---------------------------------------------------------------------------


def compute(symbol: str) -> FactorResult:
    """Compute the technical-signals score for *symbol*.

    Args:
        symbol: Ticker symbol understood by yfinance (e.g. ``"AAPL"`` or ``"RELIANCE.NS"``).

    Returns:
        :class:`~services.factors.FactorResult` with score in [-1, +1].
    """
    result = FactorResult(source="yfinance", timestamp=datetime.datetime.utcnow().isoformat())
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 20:
            result.rationale = "Insufficient price history."
            return result

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]

        # --- MA crossover ---
        sma20 = _sma(close, 20).iloc[-1]
        sma50 = _sma(close, 50)
        sma50_last = sma50.iloc[-1] if not sma50.isna().all() else None
        ma_score = 0.0
        if sma50_last and not np.isnan(sma50_last):
            pct_diff = (sma20 - sma50_last) / sma50_last
            ma_score = float(np.clip(pct_diff * 10, -1, 1))  # scale ±10% → ±1

        # --- RSI ---
        rsi_series = _rsi(close)
        rsi_val = rsi_series.iloc[-1]
        rsi_score = 0.0
        if not np.isnan(rsi_val):
            # RSI 70+ → overbought (-1), RSI 30- → oversold (+1), 50 → 0
            rsi_score = float(np.clip((50 - rsi_val) / 20, -1, 1))

        # --- Momentum (20-day return) ---
        if len(close) >= 21:
            ret20 = (close.iloc[-1] / close.iloc[-21] - 1)
            momentum_score = float(np.clip(ret20 * 5, -1, 1))  # scale ±20% → ±1
        else:
            momentum_score = 0.0

        # --- ATR volatility penalty (high vol → reduce magnitude) ---
        atr_series = _atr(high, low, close)
        atr_val = atr_series.iloc[-1]
        vol_penalty = 0.0
        if not np.isnan(atr_val) and close.iloc[-1] > 0:
            atr_pct = atr_val / close.iloc[-1]
            vol_penalty = float(np.clip(atr_pct * 5, 0, 0.3))  # up to 30% reduction

        raw_score = 0.4 * ma_score + 0.35 * rsi_score + 0.25 * momentum_score
        final_score = raw_score * (1 - vol_penalty)

        result.score = round(float(np.clip(final_score, -1, 1)), 4)
        result.rationale = (
            f"MA-cross={ma_score:.2f}, RSI={rsi_val:.1f}(score={rsi_score:.2f}), "
            f"mom20={momentum_score:.2f}, vol-penalty={vol_penalty:.2f}"
        )
        result.raw_values = {
            "sma20": round(sma20, 4),
            "sma50": round(sma50_last, 4) if sma50_last else None,
            "rsi": round(rsi_val, 2) if not np.isnan(rsi_val) else None,
            "momentum_20d": round(momentum_score, 4),
            "atr_pct": round(atr_val / close.iloc[-1], 4) if not np.isnan(atr_val) else None,
        }
    except Exception:
        logger.exception("technical_signals: error processing %s", symbol)
        result.rationale = "data unavailable"

    return result
