"""LSTM-based closing-price prediction service."""

from __future__ import annotations

import datetime as dt

from utils.logging import get_logger

logger = get_logger(__name__)


def _to_date_strings(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def _base_payload(symbol: str, lookback: int, horizon: int) -> dict:
    return {
        "symbol": symbol.upper(),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "historical": {"dates": [], "closes": []},
        "ma50": {"dates": [], "values": []},
        "forecast": {"dates": [], "closes": []},
        "metrics": {"rmse": None, "method": "unavailable"},
    }


def predict_next_30d(symbol: str, lookback: int = 50, horizon: int = 30) -> dict:
    """Predict next *horizon* trading-day closes with a lookback-window LSTM.

    Returns a JSON-serializable payload and degrades gracefully on any failure.
    """
    payload = _base_payload(symbol=symbol, lookback=lookback, horizon=horizon)
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="5y", interval="1d")
        if hist is None or hist.empty or "Close" not in hist:
            payload["error"] = "No historical close-price data available."
            return payload

        closes = hist["Close"].dropna().astype(float)
        if len(closes) < max(lookback + 20, 120):
            payload["error"] = "Not enough history for the selected lookback window."
            return payload

        recent = closes.tail(180)
        ma50 = recent.rolling(window=50).mean()

        payload["historical"] = {
            "dates": _to_date_strings(recent.index),
            "closes": [round(float(v), 4) for v in recent.values],
        }
        payload["ma50"] = {
            "dates": _to_date_strings(ma50.index),
            "values": [round(float(v), 4) if pd.notna(v) else None for v in ma50.values],
        }

        values = closes.values.reshape(-1, 1)
        min_v = float(values.min())
        max_v = float(values.max())
        denom = (max_v - min_v) or 1.0
        scaled = (values - min_v) / denom

        x_data = []
        y_data = []
        for i in range(lookback, len(scaled)):
            x_data.append(scaled[i - lookback:i, 0])
            y_data.append(scaled[i, 0])
        x = np.array(x_data, dtype="float32").reshape((-1, lookback, 1))
        y = np.array(y_data, dtype="float32")
        if len(x) < 10:
            payload["error"] = "Not enough training windows."
            return payload

        split = max(int(len(x) * 0.8), 1)
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
        if len(x_test) == 0:
            x_test, y_test = x[-1:], y[-1:]

        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import LSTM, Dense

            tf.random.set_seed(42)
            model = Sequential(
                [
                    LSTM(32, input_shape=(lookback, 1)),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(x_train, y_train, epochs=8, batch_size=16, verbose=0)

            pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
            pred = pred_scaled * denom + min_v
            actual = y_test.reshape(-1) * denom + min_v
            rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

            window = scaled[-lookback:].reshape(lookback, 1).copy()
            future_scaled = []
            for _ in range(horizon):
                p = float(model.predict(window.reshape(1, lookback, 1), verbose=0)[0][0])
                future_scaled.append(p)
                window = np.vstack([window[1:], [[p]]])

            method = "lstm"
        except Exception as exc:
            logger.exception("LSTM unavailable for %s", symbol)
            payload["error"] = f"LSTM unavailable; used moving-average fallback ({exc})."
            rmse = None
            last = float(closes.iloc[-1])
            trend = float(closes.tail(50).diff().mean() or 0.0)
            future_scaled = [((last + trend * (i + 1)) - min_v) / denom for i in range(horizon)]
            method = "moving-average-fallback"

        future = [(float(v) * denom) + min_v for v in future_scaled]
        last_date = closes.index[-1].date()
        future_dates = []
        day = last_date
        while len(future_dates) < horizon:
            day = day + dt.timedelta(days=1)
            if day.weekday() < 5:
                future_dates.append(day)

        payload["forecast"] = {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "closes": [round(float(v), 4) for v in future],
        }
        payload["metrics"] = {
            "rmse": round(rmse, 4) if rmse is not None else None,
            "method": method,
        }
        return payload

    except Exception as exc:
        logger.exception("predict_next_30d failed for %s", symbol)
        payload["error"] = f"Prediction failed: {exc}"
        return payload

