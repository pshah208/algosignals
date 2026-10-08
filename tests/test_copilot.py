"""Offline Copilot tests using the installed SDK's signatures and request types."""

import asyncio
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import create_autospec

import pytest

from copilot import CopilotClient, CopilotSession, GetAuthStatusResponse, ModelInfo, Tool, ToolSet, UriRuntimeConnection
from copilot.generated.session_events import (
    PermissionRequestCustomTool,
    PermissionRequestMcp,
    PermissionRequestRead,
    PermissionRequestShell,
    PermissionRequestUrl,
    PermissionRequestWrite,
)
from copilot.session import PermissionDecisionApproveOnce, PermissionDecisionUserNotAvailable

from services.llm import copilot_client as bridge
from services.llm import llm_client as llm


@pytest.fixture(autouse=True)
def offline_sdk(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Copilot tests must not start a CLI or access the network")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    factory = create_autospec(CopilotClient)
    client = factory.return_value
    session = create_autospec(CopilotSession, instance=True)
    session.session_id = "offline-session-id"
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None
    session.send_and_wait.return_value = SimpleNamespace(data=SimpleNamespace(content="answer"))
    client.create_session.return_value = session
    client.delete_session.return_value = None
    client.get_auth_status.return_value = GetAuthStatusResponse(
        isAuthenticated=True, login="private-login", statusMessage="private-status"
    )
    client.list_models.return_value = [
        ModelInfo(id="model-a", name="Model A", capabilities=None),
        ModelInfo(id="model-b", name="Model B", capabilities=None),
    ]
    monkeypatch.setattr("copilot.CopilotClient", factory)
    return SimpleNamespace(factory=factory, client=client, session=session)


@pytest.fixture
def runtime_factory(monkeypatch, offline_sdk):
    # Avoid actual temporary directories, while exercising the real thread/loop.
    instances = []

    def workspace(**kwargs):
        return SimpleNamespace(name=f"{bridge.__file__}-workspace-{len(instances)}", cleanup=lambda: None)

    def create(token=None):
        instance = bridge.CopilotRuntime(token=token)
        instances.append(instance)
        return instance

    monkeypatch.setattr(bridge.tempfile, "TemporaryDirectory", workspace)
    yield create
    for instance in instances:
        instance.close()
        assert not instance._thread.is_alive()
        instance._loop.close()


@pytest.fixture
def runtime(runtime_factory):
    return runtime_factory()


@pytest.fixture
def research_tools(monkeypatch):
    module = ModuleType("services.research_service")
    tools = [Tool(name="saved_signal", description="Read saved evidence")]
    servers = {"quotes": {"type": "http", "url": "https://example.invalid", "tools": ["quote"]}}
    module.make_research_tools = lambda symbol: tools
    module.tradingview_servers = lambda: servers
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return tools, servers


def test_chat_uses_sdk_async_session_contract_and_reuses_client(runtime, offline_sdk, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-models-token")
    monkeypatch.setenv("GH_TOKEN", "test-cli-token")
    monkeypatch.setenv("COPILOT_TEST_PRESERVED", "yes")
    messages = [{"role": "system", "content": "Be precise"}, {"role": "user", "content": "Hi"}]
    assert runtime.chat(messages, "model-a") == "answer"
    assert runtime.chat(messages, "model-b") == "answer"
    offline_sdk.factory.assert_called_once()
    offline_sdk.client.start.assert_awaited_once()
    options = offline_sdk.factory.call_args.kwargs
    assert options.get("use_logged_in_user", True) is True
    assert "github_token" not in options
    assert options.get("mode", "copilot-cli") == "copilot-cli"
    assert options["log_level"] == "error"
    assert options["working_directory"] == runtime._workspace.name
    assert "GITHUB_TOKEN" not in options["env"]
    assert "GH_TOKEN" not in options["env"]
    assert options["env"]["COPILOT_TEST_PRESERVED"] == "yes"
    assert [call.kwargs["model"] for call in offline_sdk.client.create_session.await_args_list] == [
        "model-a", "model-b"
    ]
    offline_sdk.session.send_and_wait.assert_awaited_with("system: Be precise\n\nuser: Hi", timeout=45)
    assert offline_sdk.session.__aenter__.await_count == 2
    assert offline_sdk.session.__aexit__.await_count == 2
    assert offline_sdk.client.delete_session.await_count == 2
    offline_sdk.client.delete_session.assert_awaited_with("offline-session-id")


def test_pure_chat_disables_all_tools_and_workspace_discovery(runtime, offline_sdk):
    runtime.chat([{"role": "user", "content": "Hi"}], "model-a")
    options = offline_sdk.client.create_session.call_args.kwargs
    assert options["available_tools"] == []
    assert options["tools"] == []
    assert options["mcp_servers"] == {}
    assert options["hooks"] == {}
    for flag in (
        "enable_config_discovery", "enable_on_demand_instruction_discovery",
        "enable_file_hooks", "enable_host_git_operations", "enable_skills", "enable_session_store",
    ):
        assert options[flag] is False
    assert options["skip_custom_instructions"] is True
    assert options["excluded_builtin_agents"] == ["*"]
    assert options["custom_agents"] == []
    assert options["infinite_sessions"] == {"enabled": False}
    assert options["system_message"]["mode"] == "replace"
    assert "not investment advice" in options["system_message"]["content"].lower()
    request = PermissionRequestCustomTool(tool_description="Unexpected tool", tool_name="saved_signal")
    assert isinstance(options["on_permission_request"](request, {}), PermissionDecisionUserNotAvailable)


def test_concurrent_requests_share_one_connection(runtime, offline_sdk):
    async def start():
        await asyncio.sleep(0.02)

    offline_sdk.client.start.side_effect = start
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda model: runtime.chat([], model), ["model-a", "model-b"]))
    assert results == ["answer", "answer"]
    offline_sdk.factory.assert_called_once()
    offline_sdk.client.start.assert_awaited_once()
    assert offline_sdk.client.create_session.await_count == 2


@pytest.mark.parametrize("permission_request", [
    PermissionRequestShell(
        can_offer_session_approval=True, commands=[], full_command_text="rm -rf project",
        has_write_file_redirection=True, intention="Run command", possible_paths=[], possible_urls=[],
        request_sandbox_bypass=True,
    ),
    PermissionRequestWrite(
        can_offer_session_approval=True, diff="+unsafe", file_name="config.py", intention="Write",
    ),
    PermissionRequestRead(intention="Read credential", path="/private/credential"),
    PermissionRequestUrl(intention="Send data", url="https://example.invalid"),
    SimpleNamespace(kind="memory"),
    SimpleNamespace(kind="hook"),
    SimpleNamespace(kind="extension-management"),
    SimpleNamespace(kind="unknown"),
])
def test_dangerous_and_unknown_permissions_rejected(permission_request, research_tools):
    tools, servers = research_tools
    handler = bridge.CopilotRuntime._session_options(tools, servers)["on_permission_request"]
    assert isinstance(handler(permission_request, {}), PermissionDecisionUserNotAvailable)


@pytest.mark.parametrize("kind,name,server,managed,approved", [
    ("custom-tool", "saved_signal", None, False, True),
    ("custom-tool", "shell", None, False, False),
    ("custom-tool", "saved_signal", None, True, False),
    ("mcp", "quote", "quotes", False, True),
    ("mcp", "write", "quotes", False, False),
    ("mcp", "quote", "unknown", False, False),
    ("mcp", "quote", "quotes", True, False),
    ("mcp", "quotes-quote", "quotes", False, False),
])
def test_only_exact_allowlisted_custom_and_mcp_permissions_approved(
    kind, name, server, managed, approved, research_tools
):
    tools, servers = research_tools
    options = bridge.CopilotRuntime._session_options(tools, servers)
    assert options["available_tools"] == ["custom:saved_signal", "mcp:quotes-quote"]
    if kind == "custom-tool":
        request = PermissionRequestCustomTool(
            tool_description="Read evidence", tool_name=name, managed_approval_required=managed
        )
    else:
        request = PermissionRequestMcp(
            read_only=True, server_name=server, tool_name=name, tool_title="Read quote",
            managed_approval_required=managed,
        )
    decision = options["on_permission_request"](request, {})
    expected = PermissionDecisionApproveOnce if approved else PermissionDecisionUserNotAvailable
    assert isinstance(decision, expected)


def test_research_hook_enforces_allowlist_and_eight_call_budget(runtime, offline_sdk, research_tools):
    decisions = []

    async def respond(prompt, *, timeout):
        options = offline_sdk.client.create_session.call_args.kwargs
        hook = options["hooks"]["on_pre_tool_use"]
        assert "Research AAPL" in prompt
        assert "Question: Explain the evidence" in prompt
        assert timeout == 45
        for name in ["shell", "quotes-write", "saved_signal", "quotes-quote"] + ["saved_signal"] * 8:
            decisions.append(hook({
                "sessionId": "offline-session", "timestamp": datetime.now(timezone.utc),
                "workingDirectory": runtime._workspace.name, "toolName": name, "toolArgs": {},
            }, {}))
        return SimpleNamespace(data=SimpleNamespace(content="Evidence-based answer"))

    offline_sdk.session.send_and_wait.side_effect = respond
    result = runtime.research("AAPL", "Explain the evidence", "model-a")
    assert [d["permissionDecision"] for d in decisions] == ["deny", "deny"] + ["allow"] * 8 + ["deny"] * 2
    assert result == {
        "answer": "Evidence-based answer", "model": "model-a",
        "tools_used": ["saved_signal", "quotes-quote"] + ["saved_signal"] * 6,
    }
    decisions.clear()
    assert len(runtime.research("AAPL", "Explain the evidence", "model-b")["tools_used"]) == 8
    assert decisions[2]["permissionDecision"] == "allow"
    assert offline_sdk.client.delete_session.await_count == 2
    offline_sdk.client.delete_session.assert_awaited_with("offline-session-id")


def test_empty_chat_and_research_responses(runtime, offline_sdk, research_tools):
    offline_sdk.session.send_and_wait.return_value = None
    assert runtime.chat([{"role": "user", "content": "Hi"}], "model-a") is None
    with pytest.raises(RuntimeError, match="No research response"):
        runtime.research("AAPL", "Explain", "model-a")
    assert offline_sdk.session.__aexit__.await_count == 2
    assert offline_sdk.client.delete_session.await_count == 2


def test_session_context_closes_when_generation_fails(runtime, offline_sdk):
    offline_sdk.session.send_and_wait.side_effect = RuntimeError("SDK failure")
    with pytest.raises(RuntimeError, match="SDK failure"):
        runtime.chat([{"role": "user", "content": "Hi"}], "model-a")
    offline_sdk.session.__aexit__.assert_awaited_once()
    offline_sdk.client.delete_session.assert_awaited_once_with("offline-session-id")


def test_metadata_uses_typed_sdk_responses_caches_and_refreshes(runtime, offline_sdk):
    first = runtime.metadata()
    assert first == {"authenticated": True, "available": ["model-a", "model-b"]}
    first["authenticated"] = False
    assert runtime.metadata()["authenticated"] is True
    offline_sdk.client.get_auth_status.assert_awaited_once()
    offline_sdk.client.list_models.assert_awaited_once()
    runtime._metadata_at -= 61
    offline_sdk.client.list_models.return_value = []
    assert runtime.metadata() == {"authenticated": True, "available": []}
    assert offline_sdk.client.get_auth_status.await_count == 2
    assert offline_sdk.client.list_models.await_count == 2
    offline_sdk.factory.assert_called_once()


def test_unauthenticated_metadata_does_not_list_models(runtime, offline_sdk):
    offline_sdk.client.get_auth_status.return_value = GetAuthStatusResponse(isAuthenticated=False)
    assert runtime.metadata() == {"authenticated": False, "available": []}
    offline_sdk.client.list_models.assert_not_awaited()


@pytest.mark.parametrize("method", ["get_auth_status", "list_models"])
def test_metadata_discovery_failures_are_cached(runtime, offline_sdk, method):
    getattr(offline_sdk.client, method).side_effect = RuntimeError("Unavailable")
    assert runtime.metadata() == {"authenticated": False, "available": []}
    assert runtime.metadata() == {"authenticated": False, "available": []}
    getattr(offline_sdk.client, method).assert_awaited_once()


def test_missing_sdk_metadata_and_chat_fall_back_without_http(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "copilot", None)
    assert runtime.metadata() == {"authenticated": False, "available": []}
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", "copilot")
    monkeypatch.setattr(bridge, "get_copilot_runtime", lambda: runtime)
    client = llm.get_llm_client("model-a")
    assert client._chat([{"role": "user", "content": "Hi"}]) is None
    assert client.score_news_sentiment("AAPL", ["Headline"])["score"] == 0.0
    rationale = client.generate_rationale("AAPL", {}, {}, 0.0, "HOLD")
    assert "AAPL" in rationale
    assert "not investment advice" in rationale.lower()


def test_failed_start_force_stops_client_and_allows_retry(runtime, offline_sdk):
    offline_sdk.client.start.side_effect = RuntimeError("Start failed")
    with pytest.raises(RuntimeError, match="Start failed"):
        runtime.chat([], "model-a")
    offline_sdk.client.force_stop.assert_awaited_once()
    assert runtime._client is None
    offline_sdk.client.start.side_effect = None
    assert runtime.chat([], "model-a") == "answer"
    assert offline_sdk.factory.call_count == 2


def test_close_stops_connected_client(runtime, offline_sdk):
    runtime.chat([], "model-a")
    runtime.close()
    offline_sdk.client.stop.assert_awaited_once()
    assert not runtime._thread.is_alive()


def test_singleton_runtime_is_reused(monkeypatch):
    factory = create_autospec(bridge.CopilotRuntime)
    register = create_autospec(bridge.atexit.register)
    monkeypatch.setattr(bridge, "_runtime", None)
    monkeypatch.setattr(bridge, "CopilotRuntime", factory)
    monkeypatch.setattr(bridge.atexit, "register", register)
    assert bridge.get_copilot_runtime() is bridge.get_copilot_runtime()
    factory.assert_called_once_with()
    register.assert_called_once_with(factory.return_value.close)


def test_copilot_provider_routes_without_token_or_http_and_models_are_isolated(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", "copilot")
    monkeypatch.setattr(llm.settings, "COPILOT_MODEL", "model-a")
    monkeypatch.setattr(llm.settings, "GITHUB_TOKEN", "")
    monkeypatch.setattr(llm.settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(llm, "_client", None)
    backend = create_autospec(bridge.CopilotRuntime, instance=True)
    backend.chat.return_value = "Copilot response"
    monkeypatch.setattr(bridge, "get_copilot_runtime", lambda: backend)
    default = llm.get_llm_client()
    selected = llm.get_llm_client("model-b")
    other = llm.get_llm_client("model-a")
    assert default.enabled and selected.enabled
    assert default.model == "model-a"
    assert selected is not default and selected is not other
    selected.set_model("model-c")
    assert default.model == other.model == "model-a"
    messages = [{"role": "user", "content": "Hi"}]
    assert selected._chat(messages) == "Copilot response"
    backend.chat.assert_called_once_with(messages, "model-c")
    assert llm.get_llm_client() is default


def test_non_copilot_provider_preserves_http_routing(monkeypatch):
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", "github_models")
    backend = create_autospec(bridge.get_copilot_runtime)
    monkeypatch.setattr(bridge, "get_copilot_runtime", backend)
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"message": {"content": "HTTP response"}}]},
    )
    post = create_autospec(llm.httpx.post, return_value=response)
    monkeypatch.setattr(llm.httpx, "post", post)
    client = llm.LLMClient(token="offline-test-token", base_url="https://example.invalid/", model="model-a")
    assert client._chat([{"role": "user", "content": "Hi"}]) == "HTTP response"
    assert post.call_args.args == ("https://example.invalid/chat/completions",)
    assert post.call_args.kwargs["json"]["model"] == "model-a"
    backend.assert_not_called()


def test_request_timeout_cancels_work_and_releases_slot(runtime):
    async def stalled(client):
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        runtime._call(stalled, timeout=0.01)
    assert runtime.chat([], "model-a") == "answer"


@pytest.mark.parametrize("mode", ["copilot-cli", "empty"])
@pytest.mark.parametrize("research", [False, True])
def test_real_sdk_serializes_empty_and_research_allowlists_without_starting_cli(
    research, mode, monkeypatch
):
    from services import research_service

    monkeypatch.setattr(research_service.settings, "TRADINGVIEW_MCP_URL", "http://127.0.0.1:8000/mcp")
    tools = research_service.make_research_tools("AAPL") if research else []
    servers = research_service.tradingview_servers() if research else {}
    options = bridge.CopilotRuntime._session_options(tools, servers)
    captured = {}
    updated = {}

    class OfflineRpc:
        async def request(self, method, payload, **kwargs):
            if method == "session.options.update":
                updated.update(payload)
                return {"success": True}
            assert method == "session.create"
            captured.update(payload)
            response = {"sessionId": payload["sessionId"]}
            kwargs["on_response_inline"](response)
            return response

    async def create():
        # A URI connection bypasses CLI discovery; the injected transport never connects.
        client = CopilotClient(
            connection=UriRuntimeConnection(url="http://localhost:12345"),
            mode=mode,
        )
        client._client = OfflineRpc()
        return await client.create_session(model="model-a", **options)

    session = asyncio.run(create())
    assert session.session_id == captured["sessionId"]
    assert "availableTools" in captured
    assert captured["availableTools"] == options["available_tools"]
    assert captured["enableConfigDiscovery"] is False
    assert captured["enableHostGitOperations"] is False
    assert captured["enableFileHooks"] is False
    assert captured["enableSkills"] is False
    assert captured["requestPermission"] is True
    expected_updates = {"sessionId": session.session_id, "skipCustomInstructions": True}
    if mode == "empty":
        expected_updates.update({
            "coauthorEnabled": False, "customAgentsLocalOnly": True,
            "includedBuiltinSkills": [], "installedPlugins": [], "manageScheduleEnabled": False,
        })
    assert updated == expected_updates
    if research:
        assert captured["mcpServers"]["tradingview"]["tools"] == list(research_service.TRADINGVIEW_TOOLS)
        assert {tool["name"] for tool in captured["tools"]} == {
            "saved_signal_evidence", "current_price"
        }
        assert set(captured["availableTools"]) == {
            "custom:saved_signal_evidence", "custom:current_price",
            *(f"mcp:tradingview-{name}" for name in research_service.TRADINGVIEW_TOOLS),
        }
        assert all(callable(tool.handler) for tool in tools)
    else:
        # SDK must send an explicit empty allowlist, not omit it as a falsey value.
        assert captured["availableTools"] == []
        assert "tools" not in captured
        assert "mcpServers" not in captured


def test_actual_research_tools_and_mcp_names_use_scoped_allowlist(runtime, offline_sdk, monkeypatch):
    from services import research_service

    monkeypatch.setattr(research_service.settings, "TRADINGVIEW_MCP_URL", "http://localhost:8000/mcp")
    accepted = [
        "saved_signal_evidence", "current_price",
        *(f"tradingview-{name}" for name in research_service.TRADINGVIEW_TOOLS),
    ]
    rejected = ["yahoo_price", "other-yahoo_price", "tradingview-shell", "mcp__tradingview__yahoo_price"]

    async def respond(prompt, *, timeout):
        options = offline_sdk.client.create_session.call_args.kwargs
        expected = (
            ToolSet().add_custom("saved_signal_evidence").add_custom("current_price")
        )
        for name in research_service.TRADINGVIEW_TOOLS:
            expected.add_mcp(f"tradingview-{name}")
        assert set(options["available_tools"]) == set(expected.to_list())
        hook = options["hooks"]["on_pre_tool_use"]
        for name in rejected:
            assert hook({"toolName": name}, {})["permissionDecision"] == "deny"
        for name in accepted:
            assert hook({"toolName": name}, {})["permissionDecision"] == "allow"
        permissions = options["on_permission_request"]
        for name in research_service.TRADINGVIEW_TOOLS:
            request = PermissionRequestMcp(
                read_only=True, server_name="tradingview", tool_name=name, tool_title=name,
            )
            assert isinstance(permissions(request, {}), PermissionDecisionApproveOnce)
        return SimpleNamespace(data=SimpleNamespace(content="Scoped evidence"))

    offline_sdk.session.send_and_wait.side_effect = respond
    result = runtime.research("AAPL", "Explain", "model-a")
    assert result["tools_used"] == accepted


def test_browser_token_runtime_disables_operator_auth_and_isolates_sdk_storage(
    runtime_factory, offline_sdk, monkeypatch
):
    monkeypatch.setenv("GITHUB_TOKEN", "host-token")
    monkeypatch.setenv("GH_TOKEN", "host-cli-token")
    runtime = runtime_factory(token="offline-browser-oauth")
    assert runtime.chat([], "model-a") == "answer"
    options = offline_sdk.factory.call_args.kwargs
    assert options["github_token"] == "offline-browser-oauth"
    assert options["use_logged_in_user"] is False
    assert options["mode"] == "empty"
    assert options["base_directory"] == runtime._workspace.name
    assert options["working_directory"] == runtime._workspace.name
    assert "GITHUB_TOKEN" not in options["env"] and "GH_TOKEN" not in options["env"]
    assert runtime.metadata() == {"authenticated": True, "available": ["model-a", "model-b"]}
    assert "offline-browser-oauth" not in repr(runtime.metadata())
    runtime.close()
    assert runtime._token is None
    assert runtime._loop.is_closed()
    with pytest.raises(RuntimeError, match="expired"):
        runtime.chat([], "model-a")
    offline_sdk.client.stop.assert_awaited_once()


def test_browser_credentials_and_metadata_are_not_shared_between_runtimes(
    runtime_factory, offline_sdk
):
    clients = [create_autospec(CopilotClient, instance=True) for _ in range(2)]
    for index, client in enumerate(clients):
        client.get_auth_status.return_value = GetAuthStatusResponse(isAuthenticated=True)
        client.list_models.return_value = [
            ModelInfo(id=f"user-{index}-model", name="Private model", capabilities=None)
        ]
    offline_sdk.factory.side_effect = clients
    first = runtime_factory(token="offline-user-one")
    second = runtime_factory(token="offline-user-two")
    assert first.metadata()["available"] == ["user-0-model"]
    assert second.metadata()["available"] == ["user-1-model"]
    assert first.metadata()["available"] == ["user-0-model"]
    assert first._workspace.name != second._workspace.name
    assert [call.kwargs["github_token"] for call in offline_sdk.factory.call_args_list] == [
        "offline-user-one", "offline-user-two"
    ]
    for client in clients:
        client.get_auth_status.assert_awaited_once()
        client.list_models.assert_awaited_once()


@pytest.mark.parametrize("oauth_configured", [False, True])
def test_request_owned_runtime_never_uses_operator_client(monkeypatch, oauth_configured):
    from flask import Flask

    auth = ModuleType("services.github_auth")
    user_runtime = create_autospec(bridge.CopilotRuntime, instance=True)
    auth.get_user_runtime = lambda: user_runtime
    monkeypatch.setitem(sys.modules, auth.__name__, auth)
    monkeypatch.setattr(bridge.settings, "GITHUB_CLIENT_ID", "offline-client" if oauth_configured else "")
    operator = create_autospec(bridge.get_copilot_runtime)
    monkeypatch.setattr(bridge, "get_copilot_runtime", operator)
    monkeypatch.setattr(llm.settings, "LLM_PROVIDER", "copilot")
    with Flask(__name__).test_request_context():
        assert bridge.get_request_copilot_runtime() is user_runtime
        selected = llm.get_llm_client("model-a")
        user_runtime.chat.return_value = "Private response"
        assert selected._chat([]) == "Private response"
        user_runtime.chat.assert_called_once_with([], "model-a")
    operator.assert_not_called()


def test_configured_browser_oauth_without_user_rejects_operator_fallback(monkeypatch):
    from flask import Flask

    auth = ModuleType("services.github_auth")
    auth.get_user_runtime = lambda: None
    monkeypatch.setitem(sys.modules, auth.__name__, auth)
    monkeypatch.setattr(bridge.settings, "GITHUB_CLIENT_ID", "offline-client")
    operator = create_autospec(bridge.get_copilot_runtime)
    monkeypatch.setattr(bridge, "get_copilot_runtime", operator)
    with Flask(__name__).test_request_context():
        with pytest.raises(RuntimeError, match="Sign in"):
            bridge.get_request_copilot_runtime()
    operator.assert_not_called()


def test_outside_request_can_still_use_trusted_operator_runtime(monkeypatch):
    monkeypatch.setattr(bridge.settings, "GITHUB_CLIENT_ID", "offline-client")
    operator = create_autospec(bridge.get_copilot_runtime)
    monkeypatch.setattr(bridge, "get_copilot_runtime", operator)
    assert bridge.get_request_copilot_runtime() is operator.return_value
    operator.assert_called_once_with()


def test_explicit_browser_llm_runtime_remains_isolated_without_models_token(monkeypatch):
    operator = create_autospec(bridge.get_copilot_runtime)
    monkeypatch.setattr(bridge, "get_copilot_runtime", operator)
    first_runtime = create_autospec(bridge.CopilotRuntime, instance=True)
    second_runtime = create_autospec(bridge.CopilotRuntime, instance=True)
    first_runtime.chat.return_value = "First user's answer"
    second_runtime.chat.return_value = "Second user's answer"
    first = llm.LLMClient(provider="copilot", token="", model="model-a", runtime=first_runtime)
    second = llm.LLMClient(provider="copilot", token="", model="model-b", runtime=second_runtime)
    assert first.enabled and second.enabled
    first.set_model("model-c")
    assert first._chat([]) == "First user's answer"
    assert second._chat([]) == "Second user's answer"
    first_runtime.chat.assert_called_once_with([], "model-c")
    second_runtime.chat.assert_called_once_with([], "model-b")
    assert first.last_call_succeeded and second.last_call_succeeded
    first_runtime.chat.side_effect = RuntimeError("Expired browser session")
    assert first._chat([]) is None
    assert not first.last_call_succeeded
    assert second.last_call_succeeded
    operator.assert_not_called()


@pytest.mark.parametrize("operation", ["chat", "research"])
def test_sdk_response_timeouts_delete_session(runtime, offline_sdk, research_tools, operation):
    offline_sdk.session.send_and_wait.side_effect = TimeoutError("SDK response timeout")
    with pytest.raises(TimeoutError, match="Copilot request timed out"):
        if operation == "chat":
            runtime.chat([], "model-a")
        else:
            runtime.research("AAPL", "Explain", "model-a")
    offline_sdk.session.__aexit__.assert_awaited_once()
    offline_sdk.client.delete_session.assert_awaited_once_with("offline-session-id")


def test_outer_request_deadline_deletes_session_before_releasing_slot(runtime, offline_sdk):
    deleted = []

    async def delete(session_id):
        await asyncio.sleep(0.01)
        deleted.append(session_id)

    async def stalled(client):
        async with runtime._disposable_session(client, "model-a", runtime._session_options()):
            await asyncio.sleep(1)

    offline_sdk.client.delete_session.side_effect = delete
    with pytest.raises(TimeoutError):
        runtime._call(stalled, timeout=0.01)
    assert deleted == ["offline-session-id"]
    offline_sdk.session.__aexit__.assert_awaited_once()
    assert runtime.chat([], "model-a") == "answer"
    assert deleted == ["offline-session-id", "offline-session-id"]


def test_session_deletion_is_shielded_from_additional_cancellation(offline_sdk):
    async def exercise():
        entered = asyncio.Event()
        delete_started = asyncio.Event()
        release_delete = asyncio.Event()
        deleted = asyncio.Event()

        async def delete(session_id):
            assert session_id == "offline-session-id"
            delete_started.set()
            await release_delete.wait()
            deleted.set()

        async def operation():
            async with bridge.CopilotRuntime._disposable_session(
                offline_sdk.client, "model-a", {}
            ):
                entered.set()
                await asyncio.Event().wait()

        offline_sdk.client.delete_session.side_effect = delete
        task = asyncio.create_task(operation())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(delete_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not deleted.is_set()
        release_delete.set()
        await asyncio.wait_for(deleted.wait(), timeout=1)
        await asyncio.sleep(0)

    asyncio.run(exercise())
    offline_sdk.session.__aexit__.assert_awaited_once()
    offline_sdk.client.delete_session.assert_awaited_once_with("offline-session-id")


def test_failed_session_entry_still_deletes_created_history(runtime, offline_sdk):
    offline_sdk.session.__aenter__.side_effect = RuntimeError("Session entry failed")
    with pytest.raises(RuntimeError, match="Session entry failed"):
        runtime.chat([], "model-a")
    offline_sdk.client.delete_session.assert_awaited_once_with("offline-session-id")
    offline_sdk.session.__aexit__.assert_not_awaited()


def test_session_deletion_has_bounded_timeout(runtime, offline_sdk, monkeypatch):
    wait_for = asyncio.wait_for
    cleanup_timeouts = []
    cancelled = []

    async def short_cleanup_timeout(awaitable, timeout):
        cleanup_timeouts.append(timeout)
        return await wait_for(awaitable, timeout=0.01)

    async def stalled_delete(session_id):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(session_id)

    monkeypatch.setattr(bridge.asyncio, "wait_for", short_cleanup_timeout)
    offline_sdk.client.delete_session.side_effect = stalled_delete
    with pytest.raises(TimeoutError, match="Copilot request timed out"):
        runtime.chat([], "model-a")
    assert cleanup_timeouts == [5]
    assert cancelled == ["offline-session-id"]
    offline_sdk.client.delete_session.assert_awaited_once_with("offline-session-id")
    offline_sdk.session.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("queue,admission_timeout", [(False, 1), (True, 120)])
def test_call_uses_distinct_interactive_and_pipeline_admission_timeouts(
    runtime, monkeypatch, queue, admission_timeout
):
    slots = create_autospec(runtime._slots, instance=True)
    slots.acquire.return_value = True
    monkeypatch.setattr(runtime, "_slots", slots)

    async def operation(client):
        return "admitted"

    assert runtime._call(operation, queue=queue) == "admitted"
    slots.acquire.assert_called_once_with(timeout=admission_timeout)
    slots.release.assert_called_once_with()


@pytest.mark.parametrize("operation", ["chat", "research", "metadata"])
def test_only_signal_chat_queues_for_capacity(runtime, monkeypatch, research_tools, operation):
    call = create_autospec(runtime._call)
    call.return_value = {"authenticated": False, "available": []}
    monkeypatch.setattr(runtime, "_call", call)
    if operation == "chat":
        runtime.chat([], "model-a")
        assert call.call_args.kwargs == {"queue": True}
    elif operation == "research":
        runtime.research("AAPL", "Explain", "model-a")
        assert call.call_args.kwargs.get("queue", False) is False
    else:
        runtime.metadata()
        assert call.call_args.kwargs == {"timeout": 20}
    call.assert_called_once()


@pytest.mark.parametrize("operation", ["research", "metadata"])
def test_busy_interactive_request_does_not_wait_for_pipeline_queue(
    runtime, monkeypatch, offline_sdk, research_tools, operation
):
    slots = create_autospec(runtime._slots, instance=True)
    slots.acquire.return_value = False
    monkeypatch.setattr(runtime, "_slots", slots)
    started = time.monotonic()
    if operation == "research":
        with pytest.raises(RuntimeError, match="busy"):
            runtime.research("AAPL", "Explain", "model-a")
    else:
        assert runtime.metadata() == {"authenticated": False, "available": []}
    assert time.monotonic() - started < 1
    slots.acquire.assert_called_once_with(timeout=1)
    slots.release.assert_not_called()
    offline_sdk.factory.assert_not_called()


def test_four_signal_calls_queue_past_old_one_second_admission_limit(runtime, offline_sdk):
    async def generate(prompt, *, timeout):
        await asyncio.sleep(1.1)
        return SimpleNamespace(data=SimpleNamespace(content="queued answer"))

    offline_sdk.session.send_and_wait.side_effect = generate
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda model: runtime.chat([], model), ["model-a"] * 4))
    assert answers == ["queued answer"] * 4
    offline_sdk.client.start.assert_awaited_once()
    assert offline_sdk.client.create_session.await_count == 4
    assert offline_sdk.client.delete_session.await_count == 4
