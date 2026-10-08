"""Read-only evidence tools for the Copilot research agent."""

import asyncio
import re
from urllib.parse import urlsplit

from config import settings
from database.db import SessionLocal
from database.models import Recommendation, SignalRun

TRADINGVIEW_TOOLS = (
    "yahoo_price",
    "combined_analysis",
    "financial_news",
    "multi_timeframe_analysis",
)


def validate_symbol(symbol):
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9^][A-Za-z0-9.^=_-]{0,19}", symbol):
        raise ValueError("Use a valid ticker of at most 20 characters.")
    return symbol.upper()


def saved_evidence(symbol):
    db = SessionLocal()
    try:
        rec = (
            db.query(Recommendation)
            .join(SignalRun)
            .filter(Recommendation.symbol == symbol, SignalRun.status == "ok")
            .order_by(SignalRun.run_at.desc(), Recommendation.id.desc())
            .first()
        )
        return rec.to_dict() if rec else {"symbol": symbol, "error": "No completed signal evidence. Run signals first."}
    finally:
        db.close()


def make_research_tools(symbol):
    """Bind local tools to the selected ticker; models cannot choose arbitrary queries."""
    from copilot import define_tool
    from pydantic import BaseModel, ConfigDict

    class NoArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def evidence(_params, _invocation):
        return await asyncio.to_thread(saved_evidence, symbol)

    async def price(_params, _invocation):
        from services.signal_service import get_current_price

        return await asyncio.to_thread(get_current_price, symbol)

    return [
        define_tool(
            name="saved_signal_evidence",
            description=f"Read the latest completed AlgoSignals recommendation and dated factor evidence for {symbol}.",
            params_type=NoArguments,
            handler=evidence,
            defer="never",
        ),
        define_tool(
            name="current_price",
            description=f"Read the latest best-effort price for {symbol}. Cache freshness is at most 60 seconds.",
            params_type=NoArguments,
            handler=price,
            defer="never",
        ),
    ]


def tradingview_servers():
    """Connect only to an explicitly configured local server, never a browser-supplied URL."""
    url = settings.TRADINGVIEW_MCP_URL
    if not url:
        return {}
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("TRADINGVIEW_MCP_URL must be a loopback HTTP URL without credentials.")
    return {
        "tradingview": {
            "type": "http",
            "url": url,
            "tools": list(TRADINGVIEW_TOOLS),
            "timeout": 15000,
        }
    }
