"""OpenAI-compatible LLM client defaulting to GitHub Models.

The client is **optional**: when no token is configured it falls back to
heuristic/templated outputs so the rest of the app continues to work.
"""

import json
import math
import threading
from typing import Any

import httpx

from config import settings
from services.llm.prompts import (
    FALLBACK_RATIONALE_TEMPLATE,
    NEWS_SENTIMENT_PROMPT,
    RATIONALE_PROMPT,
)
from utils.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 30  # seconds


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat-completion endpoint.

    Args:
        base_url: Inference endpoint base URL (default: GitHub Models).
        model: Model identifier to use.
        token: ****** If empty, all methods return graceful fallbacks.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        token: str | None = None,
        provider: str | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.provider = provider or settings.LLM_PROVIDER
        self._runtime = runtime
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or (
            settings.COPILOT_MODEL if self.provider == "copilot" else settings.LLM_MODEL
        )
        self.token = token if token is not None else settings.llm_token
        self.enabled = self.provider == "copilot" or bool(self.token)
        self._call_state = threading.local()

    @property
    def last_call_succeeded(self) -> bool:
        return getattr(self._call_state, "succeeded", False)

    def set_model(self, model: str) -> None:
        """Switch the active model identifier used for subsequent requests."""
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, str]]) -> str | None:
        """Send a chat-completion request and return the assistant content."""
        self._call_state.succeeded = False
        if not self.enabled:
            return None
        if self.provider == "copilot":
            from services.llm.copilot_client import get_copilot_runtime

            try:
                content = (self._runtime or get_copilot_runtime()).chat(messages, self.model)
                self._call_state.succeeded = bool(content)
                return content
            except Exception:
                logger.warning("Copilot unavailable; using heuristic fallback.")
                return None
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                return None
            self._call_state.succeeded = bool(content)
            return content
        except Exception:
            logger.exception("LLM call to %s failed", url)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_news_sentiment(self, symbol: str, headlines: list[str]) -> dict[str, Any]:
        """Return a sentiment score and summary for a list of news headlines.

        Args:
            symbol: Stock ticker.
            headlines: List of news headline strings.

        Returns:
            Dict with keys ``score`` (float -1..+1) and ``summary`` (str).
            Falls back to ``{"score": 0.0, "summary": "LLM unavailable"}`` on failure.
        """
        if not headlines:
            return {"score": 0.0, "summary": "No headlines available.", "available": False}

        prompt = NEWS_SENTIMENT_PROMPT.format(
            symbol=symbol,
            headlines="\n".join(f"- {h}" for h in headlines[:20]),
        )
        content = self._chat([{"role": "user", "content": prompt}])
        if content is None:
            return {"score": 0.0, "summary": "LLM unavailable.", "available": False}

        # Strip markdown fences if the model wraps the JSON
        clean = content.strip()
        if clean.startswith("```") and clean.endswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(clean)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")
            score = float(parsed.get("score", 0.0))
            if not math.isfinite(score):
                raise ValueError("Expected a finite score")
            score = max(-1.0, min(1.0, score))
            summary = str(parsed.get("summary", ""))
            return {"score": score, "summary": summary, "available": True}
        except (ValueError, TypeError, KeyError):
            logger.warning("Failed to parse LLM sentiment response.")
            return {"score": 0.0, "summary": "Could not parse LLM response.", "available": False}

    def generate_rationale(
        self,
        symbol: str,
        factor_scores: dict[str, float],
        factor_notes: dict[str, str],
        composite: float,
        action: str,
    ) -> str:
        """Generate a human-readable rationale for a recommendation.

        Args:
            symbol: Stock ticker.
            factor_scores: Mapping of factor name → score.
            factor_notes: Mapping of factor name → short rationale note.
            composite: Composite score.
            action: BUY / SELL / HOLD.

        Returns:
            Plain-English rationale string.
        """
        if not self.enabled:
            self._call_state.succeeded = False
            return self._fallback_rationale(symbol, factor_scores, composite, action)

        prompt = RATIONALE_PROMPT.format(
            symbol=symbol,
            technical=factor_scores.get("technical", 0.0),
            technical_note=factor_notes.get("technical", "n/a"),
            news=factor_scores.get("news", 0.0),
            news_note=factor_notes.get("news", "n/a"),
            events=factor_scores.get("events", 0.0),
            events_note=factor_notes.get("events", "n/a"),
            financials=factor_scores.get("financials", 0.0),
            financials_note=factor_notes.get("financials", "n/a"),
            earnings=factor_scores.get("earnings", 0.0),
            earnings_note=factor_notes.get("earnings", "n/a"),
            composite=composite,
            action=action,
        )
        content = self._chat([{"role": "user", "content": prompt}])
        if content:
            return content.strip()
        return self._fallback_rationale(symbol, factor_scores, composite, action)

    @staticmethod
    def _fallback_rationale(
        symbol: str,
        factor_scores: dict[str, float],
        composite: float,
        action: str,
    ) -> str:
        return FALLBACK_RATIONALE_TEMPLATE.format(
            symbol=symbol,
            composite=composite,
            action=action,
            technical=factor_scores.get("technical", 0.0),
            news=factor_scores.get("news", 0.0),
            events=factor_scores.get("events", 0.0),
            financials=factor_scores.get("financials", 0.0),
            earnings=factor_scores.get("earnings", 0.0),
        )


# Module-level singleton for convenience
_client: LLMClient | None = None


def get_llm_client(model: str | None = None) -> LLMClient:
    """Return a run-local client when a model is explicitly selected."""
    if model is not None:
        runtime = None
        if settings.LLM_PROVIDER == "copilot":
            from services.llm.copilot_client import get_request_copilot_runtime

            runtime = get_request_copilot_runtime()
        return LLMClient(model=model, runtime=runtime)
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
