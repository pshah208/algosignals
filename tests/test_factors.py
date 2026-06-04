"""Tests that verify every factor module returns a valid FactorResult and
degrades gracefully when its data provider is missing or raises an exception.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from services.factors import FactorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_result(result: FactorResult) -> None:
    """Assert that *result* is a well-formed FactorResult."""
    assert isinstance(result, FactorResult), "Must return FactorResult"
    assert -1.0 <= result.score <= 1.0, f"Score out of bounds: {result.score}"
    assert isinstance(result.rationale, str), "Rationale must be a string"
    assert isinstance(result.raw_values, dict), "raw_values must be a dict"
    assert isinstance(result.source, str), "source must be a string"


# ---------------------------------------------------------------------------
# technical_signals
# ---------------------------------------------------------------------------


class TestTechnicalSignals:
    def test_graceful_degradation_when_yfinance_missing(self):
        """Returns neutral 0.0 when yfinance is not importable."""
        with patch.dict(sys.modules, {"yfinance": None}):
            from services.factors import technical_signals

            result = technical_signals.compute("AAPL")
        _valid_result(result)
        assert result.score == 0.0

    def test_graceful_degradation_on_exception(self):
        """Returns neutral 0.0 when yfinance.Ticker raises."""
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network error")
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import technical_signals

            result = technical_signals.compute("AAPL")
        _valid_result(result)
        assert result.score == 0.0

    def test_graceful_degradation_empty_history(self):
        """Returns neutral 0.0 when ticker history is empty."""
        import pandas as pd

        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import technical_signals

            result = technical_signals.compute("FAKE")
        _valid_result(result)
        assert result.score == 0.0

    def test_compute_with_realistic_data(self):
        """Returns a score in [-1, +1] with synthetic OHLCV data."""
        import pandas as pd
        import numpy as np

        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
        df = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1_000_000] * n,
            }
        )
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import technical_signals

            result = technical_signals.compute("TEST")
        _valid_result(result)
        assert result.source == "yfinance"


# ---------------------------------------------------------------------------
# news_signals
# ---------------------------------------------------------------------------


class TestNewsSignals:
    def test_graceful_degradation_no_news(self):
        """Returns neutral 0.0 when all news fetches fail."""
        with (
            patch("services.factors.news_signals._fetch_newsapi", return_value=[]),
            patch("services.factors.news_signals._fetch_rss", return_value=[]),
            patch("services.factors.news_signals._get_company_name", return_value=""),
        ):
            from services.factors import news_signals

            result = news_signals.compute("AAPL", llm_client=None)
        _valid_result(result)
        assert result.score == 0.0
        assert result.raw_values["headline_count"] == 0
        assert result.raw_values["scorer"] == "none"

    def test_lexicon_positive_headlines(self):
        """Returns positive score for bullish headlines using lexicon fallback."""
        headlines = ["Company beats earnings", "Stock surges on strong profit growth"]
        with (
            patch("services.factors.news_signals._fetch_newsapi", return_value=[]),
            patch("services.factors.news_signals._fetch_rss", return_value=headlines),
            patch("services.factors.news_signals._get_company_name", return_value="Apple Inc"),
        ):
            from services.factors import news_signals

            result = news_signals.compute("AAPL", llm_client=None)
        _valid_result(result)
        assert result.score > 0
        assert result.raw_values["source_used"] == "rss"
        assert result.raw_values["headline_count"] == len(headlines)
        assert result.raw_values["scorer"] in {"vader", "lexicon"}

    def test_lexicon_negative_headlines(self):
        """Returns negative score for bearish headlines."""
        headlines = ["Company misses earnings forecast", "Stock falls on rising losses"]
        with (
            patch("services.factors.news_signals._fetch_newsapi", return_value=[]),
            patch("services.factors.news_signals._fetch_rss", return_value=headlines),
            patch("services.factors.news_signals._get_company_name", return_value=""),
        ):
            from services.factors import news_signals

            result = news_signals.compute("AAPL", llm_client=None)
        _valid_result(result)
        assert result.score < 0

    def test_llm_client_used_when_enabled(self):
        """Calls llm_client.score_news_sentiment when LLM is enabled."""
        headlines = ["Earnings beat expectations"]
        mock_llm = MagicMock()
        mock_llm.enabled = True
        mock_llm.score_news_sentiment.return_value = {"score": 0.8, "summary": "Very bullish"}
        with (
            patch("services.factors.news_signals._fetch_newsapi", return_value=[]),
            patch("services.factors.news_signals._fetch_rss", return_value=headlines),
            patch("services.factors.news_signals._get_company_name", return_value=""),
        ):
            from services.factors import news_signals

            result = news_signals.compute("AAPL", llm_client=mock_llm)
        mock_llm.score_news_sentiment.assert_called_once()
        assert result.score == pytest.approx(0.8)
        assert result.raw_values["llm_used"] is True
        assert result.raw_values["scorer"] == "llm"

    def test_llm_failure_falls_back_to_lexicon(self):
        """Falls back to lexicon when LLM raises an exception."""
        headlines = ["Company beats earnings and surges on profit gain"]
        mock_llm = MagicMock()
        mock_llm.enabled = True
        mock_llm.score_news_sentiment.side_effect = RuntimeError("LLM down")
        with (
            patch("services.factors.news_signals._fetch_newsapi", return_value=[]),
            patch("services.factors.news_signals._fetch_rss", return_value=headlines),
            patch("services.factors.news_signals._get_company_name", return_value=""),
        ):
            from services.factors import news_signals

            result = news_signals.compute("AAPL", llm_client=mock_llm)
        _valid_result(result)
        # Lexicon fallback should produce a non-zero score for clearly bullish text
        assert result.score > 0


# ---------------------------------------------------------------------------
# events_signals
# ---------------------------------------------------------------------------


class TestEventsSignals:
    def test_graceful_degradation_when_yfinance_fails(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network error")
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import events_signals

            result = events_signals.compute("AAPL")
        _valid_result(result)
        assert result.score == 0.0

    def test_returns_valid_result_with_mock_data(self):
        import pandas as pd

        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.dividends = pd.Series(
            [0.22, 0.24], index=pd.to_datetime(["2023-09-01", "2024-03-01"])
        )
        mock_ticker.splits = pd.Series(dtype=float)
        mock_ticker.calendar = {}
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import events_signals

            result = events_signals.compute("AAPL")
        _valid_result(result)
        assert "next_earnings_days" in result.raw_values


# ---------------------------------------------------------------------------
# financials_signals
# ---------------------------------------------------------------------------


class TestFinancialsSignals:
    def test_graceful_degradation_empty_info(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import financials_signals

            result = financials_signals.compute("AAPL")
        _valid_result(result)
        assert result.score == 0.0

    def test_positive_fundamentals(self):
        """Healthy fundamentals produce a positive score."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 18.0,
            "revenueGrowth": 0.20,
            "profitMargins": 0.25,
            "debtToEquity": 30.0,  # 0.30 after /100 → low debt
        }
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import financials_signals

            result = financials_signals.compute("HEALTHY")
        _valid_result(result)
        assert result.score > 0

    def test_score_bounds(self):
        """Score is always clamped to [-1, +1]."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        # Extreme values
        mock_ticker.info = {
            "trailingPE": 200.0,
            "revenueGrowth": -0.5,
            "profitMargins": -0.3,
            "debtToEquity": 500.0,
        }
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import financials_signals

            result = financials_signals.compute("BAD")
        _valid_result(result)


# ---------------------------------------------------------------------------
# earnings_signals
# ---------------------------------------------------------------------------


class TestEarningsSignals:
    def test_graceful_degradation_when_yfinance_fails(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network error")
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import earnings_signals

            result = earnings_signals.compute("AAPL")
        _valid_result(result)
        assert result.score == 0.0

    def test_positive_eps_surprise(self):
        """Positive EPS surprise → positive score."""
        import pandas as pd

        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        # earnings_history as a DataFrame
        eh = pd.DataFrame(
            [{"epsEstimate": 1.0, "epsActual": 1.3}]
        )
        mock_ticker.earnings_history = eh
        mock_ticker.calendar = {}
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from services.factors import earnings_signals

            result = earnings_signals.compute("BEATS")
        _valid_result(result)
        assert result.score > 0
