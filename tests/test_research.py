"""Offline tests for the protected, read-only research API and tools."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from config import settings
from services.research_service import make_research_tools, tradingview_servers, validate_symbol


@pytest.fixture
def client(monkeypatch):
    from blueprints.dashboard import bp as dashboard
    from blueprints.models import bp as models
    from blueprints.research import bp as research

    monkeypatch.setattr(settings, "APP_API_KEY", "offline-test-key")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "copilot")
    app = Flask(__name__)
    app.secret_key = "offline-session"
    app.register_blueprint(dashboard)
    app.register_blueprint(models)
    app.register_blueprint(research)
    app.config["TESTING"] = True
    return app.test_client()


HEADERS = {"Authorization": "Bearer " + "offline-test-key"}


def test_research_requires_key_before_calling_sdk(client):
    with patch("services.llm.copilot_client.get_copilot_runtime") as sdk:
        assert client.post("/api/v1/research", json={"symbol": "AAPL"}).status_code == 401
        assert client.get("/api/v1/copilot/status").status_code == 401
        sdk.assert_not_called()


@pytest.mark.parametrize("key", ["", "change-me-api-key"])
def test_no_configured_key_fails_closed(client, monkeypatch, key):
    monkeypatch.setattr(settings, "APP_API_KEY", key)
    assert client.post("/api/v1/research", headers=HEADERS, json={"symbol": "AAPL"}).status_code == 503


@pytest.mark.parametrize("data", [[], None, {"symbol": "../etc"}, {"symbol": 123}, {"symbol": "AAPL", "question": 42}, {"symbol": "AAPL", "question": " "}, {"symbol": "AAPL", "question": "x" * 2001}])
def test_invalid_research_inputs(client, data):
    assert client.post("/api/v1/research", headers=HEADERS, json=data).status_code == 400


def test_research_uses_session_model_and_returns_trace(client):
    runtime = MagicMock()
    runtime.metadata.return_value = {"authenticated": True, "available": ["gpt-4.1", "other"]}
    runtime.research.return_value = {"answer": "Research only.", "tools_used": ["saved_signal_evidence"], "model": "other"}
    with patch("services.llm.copilot_client.get_copilot_runtime", return_value=runtime):
        client.post("/api/models", json={"model": "other"})
        response = client.post("/api/v1/research", headers=HEADERS, json={"symbol": "aapl", "question": " Risks? "})
    assert response.status_code == 200
    runtime.research.assert_called_once_with("AAPL", "Risks?", "other")
    assert response.json["tools_used"] == ["saved_signal_evidence"]
    assert "not investment advice" in response.json["disclaimer"]


def test_sdk_failure_is_generic(client):
    runtime = MagicMock()
    runtime.metadata.return_value = {"authenticated": True, "available": [settings.COPILOT_MODEL]}
    runtime.research.side_effect = RuntimeError("private diagnostic")
    with patch("services.llm.copilot_client.get_copilot_runtime", return_value=runtime):
        response = client.post("/api/v1/research", headers=HEADERS, json={"symbol": "AAPL"})
    assert response.status_code == 503
    assert "private diagnostic" not in response.get_data(as_text=True)


def test_models_provider_does_not_launch_copilot(client, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "models")
    with patch("services.llm.copilot_client.get_copilot_runtime") as sdk:
        assert client.post("/api/v1/research", headers=HEADERS, json={"symbol": "AAPL"}).status_code == 503
        assert client.get("/api/v1/copilot/status", headers=HEADERS).json["enabled"] is False
        sdk.assert_not_called()


def test_copilot_run_now_requires_key(client):
    with patch("services.signal_service.run_signals") as run:
        assert client.post("/run-now").status_code == 401
        run.assert_not_called()


@pytest.mark.parametrize("symbol", ["AAPL", "reliance.ns", "^GSPC", "BTC-USD", "EURUSD=X"])
def test_symbol_validation(symbol):
    assert validate_symbol(symbol) == symbol.upper()


@pytest.mark.parametrize("url", ["https://example.com/mcp", "http://example.com/mcp", "http://localhost@evil.example/mcp", "http://localhost/mcp?key=secret", "http://localhost/mcp#fragment"])
def test_mcp_rejects_nonlocal_or_credential_urls(monkeypatch, url):
    monkeypatch.setattr(settings, "TRADINGVIEW_MCP_URL", url)
    with pytest.raises(ValueError):
        tradingview_servers()


def test_mcp_is_opt_in_and_allowlisted(monkeypatch):
    monkeypatch.setattr(settings, "TRADINGVIEW_MCP_URL", "")
    assert tradingview_servers() == {}
    monkeypatch.setattr(settings, "TRADINGVIEW_MCP_URL", "http://127.0.0.1:8000/mcp")
    server = tradingview_servers()["tradingview"]
    assert server["tools"] == ["yahoo_price", "combined_analysis", "financial_news", "multi_timeframe_analysis"]
    assert "*" not in server["tools"]


def test_local_tools_are_bound_to_symbol_and_validate_arguments():
    from copilot import ToolInvocation

    async def invoke():
        tools = make_research_tools("AAPL")
        invocation = ToolInvocation(session_id="test", tool_call_id="test", tool_name="saved_signal_evidence", arguments={})
        with patch("services.research_service.saved_evidence", return_value={"symbol": "AAPL"}) as evidence:
            result = await tools[0].handler(invocation)
            evidence.assert_called_once_with("AAPL")
            assert result.result_type == "success"
            invocation.arguments = {"symbol": "MSFT"}
            rejected = await tools[0].handler(invocation)
            assert rejected.result_type == "failure"
            assert evidence.call_count == 1

    asyncio.run(invoke())
