"""News-sentiment factor.

Data sources (in priority order):
1. Google News RSS feed (free, no key required).
2. NewsAPI (optional; used as a supplemental fallback).

Sentiment scoring:
- If an LLMClient with a valid token is available → use it.
- Otherwise → VADER sentiment (if installed), then lexicon fallback.

Degrades gracefully to score=0.0 when all sources fail.
"""

import datetime
import re
import urllib.parse
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


def _get_company_name(symbol: str) -> str:
    """Best-effort company-name lookup used to improve RSS queries."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        return (info.get("shortName") or info.get("longName") or "").strip()
    except Exception:
        logger.debug("news_signals: company name lookup failed for %s", symbol, exc_info=True)
        return ""


def _fetch_rss(symbol: str, company_name: str = "") -> list[str]:
    """Fetch headlines from Google News RSS (no key required)."""
    try:
        import feedparser

        query_terms = [symbol, "stock"]
        if company_name:
            query_terms.insert(0, company_name)
        query = urllib.parse.quote_plus(" ".join(query_terms))
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:25]:
            title = (entry.get("title", "") or "").strip()
            if title and title not in headlines:
                headlines.append(title)
        return headlines
    except Exception:
        logger.exception("news_signals: RSS fetch failed for %s", symbol)
        return []


def _vader_score(text: str) -> float | None:
    """Return VADER compound sentiment score in [-1,+1] when available."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return float(analyzer.polarity_scores(text).get("compound", 0.0))
    except Exception:
        logger.debug("news_signals: VADER unavailable; using lexicon fallback", exc_info=True)
        return None


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
    source_used = "none"
    company_name = _get_company_name(symbol)

    # 1. RSS first (works without API keys)
    headlines = _fetch_rss(symbol, company_name=company_name)
    if headlines:
        source_used = "rss"

    # 2. Optional NewsAPI fallback/supplement if RSS is sparse
    if settings.NEWS_API_KEY and len(headlines) < 5:
        from_newsapi = _fetch_newsapi(symbol, settings.NEWS_API_KEY)
        if from_newsapi:
            source_used = "rss+newsapi" if headlines else "newsapi"
            for h in from_newsapi:
                if h not in headlines:
                    headlines.append(h)

    if not headlines:
        result.rationale = "No recent headlines found from RSS/NewsAPI."
        result.raw_values = {
            "headline_count": 0,
            "source_used": source_used,
            "company_name_hint": company_name,
            "llm_used": False,
            "scorer": "none",
        }
        return result

    # Sentiment scoring
    llm_used = False
    scorer = "lexicon"
    if llm_client and getattr(llm_client, "enabled", False):
        try:
            llm_result = llm_client.score_news_sentiment(symbol, headlines)
            score = llm_result.get("score", 0.0)
            summary = llm_result.get("summary", "")
            llm_used = True
            scorer = "llm"
        except Exception:
            logger.exception("news_signals: LLM scoring failed for %s", symbol)
            vader = _vader_score(" ".join(headlines))
            if vader is not None:
                score = vader
                scorer = "vader"
                summary = "LLM failed; used VADER fallback."
            else:
                score = _lexicon_score(" ".join(headlines))
                summary = "LLM failed; used lexicon fallback."
    else:
        vader = _vader_score(" ".join(headlines))
        if vader is not None:
            score = vader
            scorer = "vader"
            summary = "VADER-based sentiment (LLM disabled)."
        else:
            score = _lexicon_score(" ".join(headlines))
            summary = "Lexicon-based sentiment (LLM disabled)."

    result.score = round(max(-1.0, min(1.0, score)), 4)
    result.rationale = summary or f"Analysed {len(headlines)} headlines."
    result.raw_values = {
        "headline_count": len(headlines),
        "headlines_sample": headlines[:3],
        "source_used": source_used,
        "company_name_hint": company_name,
        "llm_used": llm_used,
        "scorer": scorer,
    }
    return result
