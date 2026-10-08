"""Tests that verify the LLM layer works correctly when no token is configured."""

from unittest.mock import MagicMock, patch

import pytest

from services.llm.llm_client import LLMClient
from services.llm.prompts import FALLBACK_RATIONALE_TEMPLATE


class TestLLMClientDisabled:
    """When no token is provided, LLMClient must degrade gracefully."""

    def setup_method(self):
        self.client = LLMClient(token="")

    def test_enabled_is_false(self):
        assert self.client.enabled is False

    def test_score_news_sentiment_returns_neutral(self):
        result = self.client.score_news_sentiment("AAPL", ["Headlines..."])
        assert result["score"] == 0.0
        assert isinstance(result["summary"], str)

    def test_score_news_sentiment_no_headlines(self):
        result = self.client.score_news_sentiment("AAPL", [])
        assert result["score"] == 0.0

    def test_generate_rationale_returns_string(self):
        rationale = self.client.generate_rationale(
            symbol="AAPL",
            factor_scores={"technical": 0.4, "news": 0.1, "events": 0.0,
                           "financials": 0.3, "earnings": 0.2},
            factor_notes={"technical": "bullish MA", "news": "positive headlines",
                          "events": "n/a", "financials": "good margins",
                          "earnings": "beat"},
            composite=0.25,
            action="BUY",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 10

    def test_rationale_contains_symbol(self):
        rationale = self.client.generate_rationale(
            symbol="TSLA",
            factor_scores={k: 0.0 for k in ("technical", "news", "events", "financials", "earnings")},
            factor_notes={k: "n/a" for k in ("technical", "news", "events", "financials", "earnings")},
            composite=0.0,
            action="HOLD",
        )
        assert "TSLA" in rationale

    def test_rationale_contains_disclaimer(self):
        rationale = self.client.generate_rationale(
            symbol="X",
            factor_scores={k: 0.0 for k in ("technical", "news", "events", "financials", "earnings")},
            factor_notes={k: "n/a" for k in ("technical", "news", "events", "financials", "earnings")},
            composite=0.0,
            action="HOLD",
        )
        # Fallback template includes "not investment advice"
        assert "not investment advice" in rationale.lower()


class TestLLMClientEnabled:
    """When a token is provided but the HTTP call fails, client must degrade."""

    def setup_method(self):
        self.client = LLMClient(token="fake-token-for-testing")

    def test_enabled_is_true(self):
        assert self.client.enabled is True

    def test_score_sentiment_http_failure_returns_neutral(self):
        """HTTP failure → neutral score, no crash."""
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("unreachable")):
            result = self.client.score_news_sentiment("AAPL", ["Headline"])
        assert result["score"] == 0.0

    def test_rationale_http_failure_returns_fallback(self):
        """HTTP failure → fallback rationale string, no crash."""
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("unreachable")):
            rationale = self.client.generate_rationale(
                symbol="AAPL",
                factor_scores={k: 0.0 for k in ("technical", "news", "events", "financials", "earnings")},
                factor_notes={k: "n/a" for k in ("technical", "news", "events", "financials", "earnings")},
                composite=0.0,
                action="HOLD",
            )
        assert isinstance(rationale, str)
        assert "AAPL" in rationale

    def test_score_sentiment_invalid_json_returns_neutral(self):
        """Malformed JSON response → neutral score, no crash."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "not valid json {{"}}]
        }
        with patch("httpx.post", return_value=mock_resp):
            result = self.client.score_news_sentiment("AAPL", ["Headline"])
        assert result["score"] == 0.0

    def test_score_sentiment_clamps_score(self):
        """LLM returning out-of-range score is clamped to [-1, +1]."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"score": 99.0, "summary": "extreme"}'}}]
        }
        with patch("httpx.post", return_value=mock_resp):
            result = self.client.score_news_sentiment("AAPL", ["Headline"])
        assert result["score"] == 1.0

    def test_authorization_uses_configured_token(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("httpx.post", return_value=mock_resp) as post:
            assert self.client._chat([{"role": "user", "content": "hello"}]) == "ok"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer " + self.client.token
        assert self.client.last_call_succeeded is True

    @pytest.mark.parametrize("content", ['[]', '{"score": null}', '{"score": NaN}', '{"score": Infinity}', '```json\n{"score": 0.3, "summary": "ok"}\n```'])
    def test_sentiment_validation(self, content):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        with patch("httpx.post", return_value=mock_resp):
            result = self.client.score_news_sentiment("AAPL", ["Headline"])
        if content.startswith("```"):
            assert result["score"] == 0.3
            assert result["available"] is True
        else:
            assert result["available"] is False
