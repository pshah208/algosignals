"""Tests for metrics/prediction pages and dashboard additions."""

import datetime
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from flask import Flask


@pytest.fixture()
def app():
    from blueprints.dashboard import bp as dashboard_bp
    from blueprints.models import bp as models_bp
    from blueprints.predict import bp as predict_bp
    from blueprints.watchlist import bp as watchlist_bp
    from blueprints.config_routes import bp as config_bp

    flask_app = Flask(__name__, template_folder="../templates")
    flask_app.secret_key = "test-secret"
    flask_app.register_blueprint(models_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(predict_bp)
    flask_app.register_blueprint(watchlist_bp)
    flask_app.register_blueprint(config_bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


class _Run:
    id = 1
    run_at = datetime.datetime(2026, 1, 1, 12, 0, 0)
    status = "ok"
    summary = "ok"

    def to_dict(self):
        return {
            "id": self.id,
            "run_at": self.run_at.isoformat(),
            "status": self.status,
            "summary": self.summary,
        }


class _Rec:
    id = 1
    run_id = 1
    symbol = "AAPL"
    exchange = "NASDAQ"
    action = "BUY"
    composite_score = 0.42
    rationale = "Test rationale."
    llm_used = False
    created_at = datetime.datetime(2026, 1, 1, 12, 0, 0)

    @property
    def factor_scores_dict(self):
        return {k: {"score": 0.1, "rationale": "ok"} for k in ("technical", "news", "events", "financials", "earnings")}

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "action": self.action,
            "composite_score": self.composite_score,
            "factor_scores": self.factor_scores_dict,
            "rationale": self.rationale,
            "llm_used": self.llm_used,
            "created_at": self.created_at.isoformat(),
        }


def test_metrics_page_available(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Scoring Metrics" in html
    assert "Composite Score & Actions" in html


def test_dashboard_includes_current_price_and_prediction_link(client):
    mock_db = MagicMock()
    q_run = MagicMock()
    q_run.order_by.return_value.first.return_value = _Run()
    q_rec = MagicMock()
    q_rec.filter_by.return_value.order_by.return_value.all.return_value = [_Rec()]
    mock_db.query.side_effect = [q_run, q_rec]

    with (
        patch("blueprints.dashboard.SessionLocal", return_value=mock_db),
        patch("blueprints.dashboard.get_current_price", return_value={"price": 123.45, "currency": "USD"}),
    ):
        resp = client.get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Current Price" in html
    assert "/predict/AAPL" in html
    assert "123.45 USD" in html


def test_predict_routes_render_and_return_json(client):
    payload = {
        "symbol": "AAPL",
        "lookback": 50,
        "horizon": 30,
        "historical": {"dates": ["2026-01-01"], "closes": [100.0]},
        "ma50": {"dates": ["2026-01-01"], "values": [99.0]},
        "forecast": {"dates": ["2026-01-02"], "closes": [101.0]},
        "metrics": {"rmse": 1.2, "method": "lstm"},
    }
    with patch("blueprints.predict.predict_next_30d", return_value=payload):
        html_resp = client.get("/predict/AAPL")
        api_resp = client.get("/predict/api/AAPL")

    assert html_resp.status_code == 200
    assert "30-Day Price Prediction: AAPL" in html_resp.get_data(as_text=True)
    assert api_resp.status_code == 200
    data = api_resp.get_json()
    assert data["forecast"]["closes"] == [101.0]


def test_lstm_predictor_graceful_when_tensorflow_missing():
    from services.prediction import lstm_predictor

    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    df = pd.DataFrame({"Close": np.linspace(100, 150, len(dates))}, index=dates)

    mock_yf = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    mock_yf.Ticker.return_value = mock_ticker

    with patch.dict(sys.modules, {"yfinance": mock_yf, "tensorflow": None}):
        result = lstm_predictor.predict_next_30d("AAPL", lookback=50, horizon=30)

    assert result["symbol"] == "AAPL"
    assert len(result["forecast"]["closes"]) == 30
    assert result["metrics"]["method"] == "moving-average-fallback"
    assert "error" in result
