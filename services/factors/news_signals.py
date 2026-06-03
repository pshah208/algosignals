"""News-sentiment factor.

Data sources (in priority order):
1. NewsAPI (requires ``NEWS_API_KEY`` env var).
2. Google News RSS feed (free, no key required).

Sentiment scoring:
- If an LLMClient with a valid token is available → use it.
- Otherwise → lightweight lexicon scoring.

Degrades gracefully to score=0.0 when all sources fail.
"""

import datetime
import re
from typing import Any

from services.factors import FactorResult
from utils.logging import get_logger

logger = get_logger(__name__)

# Simple positive/negative lexicon for fallback scoring
_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "record", "profit", "growth", "upgrade",
    "strong", "buy", "bullish", "rally", "gain", "gains", "positive", "outperform",
    "revenue", "expansion", "dividend", "raised", "raise", "increased", "momentum",
    "innovation", "partnership", "awarded", "wins", "win",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "fall", "falls", "loss", "losses", "downgrade", "weak",
    "sell", "bearish", "decline", "declines", "negative", "underperform",
    "cut", "cuts", "concern", "risk", "lawsuit", "fine", "penalty", "fraud",
    "layoff", "layoffs", "debt", "default", "warning", "recall",
}


def _lexicon_score(text: str) -> float:
    """Return a simple lexicon-based sentiment score in [-1, +1]."""
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 4)


def _fetch_newsapi(symbol: str, api_key: str) -> list[str]:
    """Fetch headlines from NewsAPI."""
    try:
        import httpx

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": api_key,
        }
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") or "" for a in articles if a.get("title")]
    except Exception:
        logger.exception("news_signals: NewsAPI fetch failed for %s", symbol)
        return []


def _fetch_rss(symbol: str) -> list[str]:
    """Fetch headlines from Google News RSS (no key required)."""
    try:
        import feedparser

        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        return [entry.get("title", "") for entry in feed.entries[:20]]
    except Exception:
        logger.exception("news_signals: RSS fetch failed for %s", symbol)
        return []


def compute(symbol: str, llm_client: Any | None = None) -> FactorResult:
    """Compute the news-sentiment score for *symbol*.

    Args:
        symbol: Ticker symbol.
        llm_client: Optional :class:`~services.llm.llm_client.LLMClient`.

    Returns:
        :class:`~services.factors.FactorResult` with score in [-1, +1].
    """
    from config import settings

    result = FactorResult(
        source="news",
        timestamp=datetime.datetime.utcnow().isoformat(),
    )

    headlines: list[str] = []

    # 1. Try NewsAPI
    if settings.NEWS_API_KEY:
        headlines = _fetch_newsapi(symbol, settings.NEWS_API_KEY)

    # 2. Fallback to RSS
    if not headlines:
        headlines = _fetch_rss(symbol)

    if not headlines:
        result.rationale = "No news data available."
        return result

    # Sentiment scoring
    llm_used = False
    if llm_client and getattr(llm_client, "enabled", False):
        try:
            llm_result = llm_client.score_news_sentiment(symbol, headlines)
            score = llm_result.get("score", 0.0)
            summary = llm_result.get("summary", "")
            llm_used = True
        except Exception:
            logger.exception("news_signals: LLM scoring failed for %s", symbol)
            score = _lexicon_score(" ".join(headlines))
            summary = "LLM failed; used lexicon fallback."
    else:
        score = _lexicon_score(" ".join(headlines))
        summary = "Lexicon-based sentiment (LLM disabled)."

    result.score = round(max(-1.0, min(1.0, score)), 4)
    result.rationale = summary or f"Analysed {len(headlines)} headlines."
    result.raw_values = {
        "headline_count": len(headlines),
        "headlines_sample": headlines[:3],
        "llm_used": llm_used,
    }
    return result
