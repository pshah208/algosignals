"""Offline browser OAuth tests; no SDK process or network may be started."""

import base64
import hashlib
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from flask import Flask, jsonify

from blueprints import auth
from services import github_auth


@pytest.fixture
def app(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OAuth tests must not start processes or access the network")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(httpx, "post", forbidden)
    monkeypatch.setattr(httpx, "get", forbidden)
    monkeypatch.setattr(auth.settings, "GITHUB_CLIENT_ID", "test-client", raising=False)
    monkeypatch.setattr(auth.settings, "GITHUB_CLIENT_SECRET", "synthetic-client-secret", raising=False)
    monkeypatch.setattr(auth.settings, "FLASK_ENV", "development")
    monkeypatch.setattr(
        auth.settings, "GITHUB_REDIRECT_URI", "http://localhost/auth/github/callback", raising=False
    )
    monkeypatch.setattr(github_auth, "_vault", github_auth.TokenVault())
    monkeypatch.setattr(github_auth, "CopilotRuntime", Mock(side_effect=lambda **kw: Mock()))
    application = Flask(__name__)
    application.secret_key = "synthetic-session-key-not-a-real-deployment"
    application.register_blueprint(auth.bp)

    @application.get("/identity")
    def identity():
        return jsonify(github_auth.get_login_info())

    yield application
    github_auth._vault.close()


@pytest.fixture
def exchange(monkeypatch):
    token = "gho_" + "synthetic-access-value"
    post = Mock(return_value=Mock(json=Mock(return_value={
        "access_token": token, "token_type": "bearer",
    })))
    get = Mock(return_value=Mock(json=Mock(return_value={"id": 123, "login": "octocat"})))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    return SimpleNamespace(post=post, get=get, token=token)


def _begin(client):
    response = client.get("/auth/github/login")
    assert response.status_code == 302
    location = urlsplit(response.location)
    assert location.scheme == "https"
    assert location.netloc == "github.com"
    assert location.path == "/login/oauth/authorize"
    return parse_qs(location.query)


def _login(client):
    params = _begin(client)
    return client.get("/auth/github/callback", query_string={
        "state": params["state"][0], "code": "synthetic-code",
    })


@pytest.mark.parametrize("key", ["dev-secret-key", "change-me-in-production", "short-secret", None])
def test_login_refuses_insecure_session_secret(app, key):
    app.secret_key = key
    assert app.test_client().get("/auth/github/login").status_code == 503
    assert not github_auth._vault._pending


def test_login_requires_oauth_credentials(app, monkeypatch):
    monkeypatch.setattr(auth.settings, "GITHUB_CLIENT_SECRET", "")
    assert app.test_client().get("/auth/github/login").status_code == 503


@pytest.mark.parametrize(("uri", "environment", "status"), [
    ("https://example.com/auth/github/callback", "production", 302),
    ("https://example.com:443/auth/github/callback", "production", 302),
    ("http://localhost/auth/github/callback", "development", 302),
    ("http://127.0.0.1:5000/auth/github/callback", "development", 302),
    ("http://[::1]:5000/auth/github/callback", "development", 302),
    ("http://localhost/auth/github/callback", "production", 503),
    ("http://127.0.0.1/auth/github/callback", "production", 503),
    ("http://example.com/auth/github/callback", "development", 503),
    ("https://" + "user" + ":pass@" + "example.com/auth/github/callback", "production", 503),
    ("https://example.com/auth/github/callback?next=evil", "production", 503),
    ("https://example.com/auth/github/callback?", "production", 503),
    ("https://example.com/auth/github/callback#fragment", "production", 503),
    ("https://example.com/auth/github/callback#", "production", 503),
    ("https://example.com/auth/github/callback\n", "production", 503),
    ("https://example.com/wrong/callback", "production", 503),
    ("https://example.com:bad/auth/github/callback", "production", 503),
    ("https:///auth/github/callback", "production", 503),
    ("javascript:alert(1)", "production", 503),
])
def test_redirect_uri_security(app, monkeypatch, uri, environment, status):
    monkeypatch.setattr(auth.settings, "GITHUB_REDIRECT_URI", uri)
    monkeypatch.setattr(auth.settings, "FLASK_ENV", environment)
    assert app.test_client().get("/auth/github/login").status_code == status


def test_session_secret_minimum_length(app):
    app.secret_key = "x" * 31
    assert app.test_client().get("/auth/github/login").status_code == 503
    app.secret_key = "x" * 32
    assert app.test_client().get("/auth/github/login").status_code == 302


def test_pkce_exchange_and_cookie_excludes_token(app, exchange):
    client = app.test_client()
    params = _begin(client)
    assert params["code_challenge_method"] == ["S256"]
    assert "scope" not in params  # No repository permissions are needed for identity.
    cookie_before = client.get_cookie("session").value
    response = client.get("/auth/github/callback", query_string={
        "state": params["state"][0], "code": "synthetic-code",
    })
    assert response.status_code == 302
    assert response.location == "/"
    assert response.headers["Cache-Control"] == "no-store"
    args, kwargs = exchange.post.call_args
    assert args == ("https://github.com/login/oauth/access_token",)
    assert kwargs["headers"] == {"Accept": "application/json"}
    assert kwargs["timeout"] == 10
    data = kwargs["data"]
    assert data["client_id"] == "test-client"
    assert data["client_secret"] == "synthetic-client-secret"
    assert data["redirect_uri"] == "http://localhost/auth/github/callback"
    assert data["code"] == "synthetic-code"
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(data["code_verifier"].encode()).digest()
    ).rstrip(b"=").decode()
    assert params["code_challenge"] == [challenge]
    assert exchange.get.call_args.args == ("https://api.github.com/user",)
    assert exchange.get.call_args.kwargs["headers"]["Authorization"] == "Bearer " + exchange.token
    assert exchange.get.call_args.kwargs["timeout"] == 10
    github_auth.CopilotRuntime.assert_called_once_with(token=exchange.token)
    decoded = app.session_interface.get_signing_serializer(app).loads(
        client.get_cookie("session").value
    )
    assert set(decoded) == {"github_session", "github_csrf"}
    assert exchange.token not in str(decoded)
    assert "synthetic-client-secret" not in str(decoded)
    assert client.get("/identity").json["login"] == "octocat"
    # Even a replay of the original signed cookie cannot restore consumed state.
    client.set_cookie("session", cookie_before)
    assert client.get("/auth/github/callback", query_string={
        "state": params["state"][0], "code": "synthetic-code",
    }).status_code == 400
    assert exchange.post.call_count == 1


@pytest.mark.parametrize("callback", [
    {"state": "wrong", "code": "code"},
    {"code": "code"},
    {"state": "", "code": "code"},
    {"error": "access_denied"},
    {},
])
def test_invalid_callback_consumes_pending_state(app, exchange, callback):
    client = app.test_client()
    state = _begin(client)["state"][0]
    assert client.get("/auth/github/callback", query_string=callback).status_code == 400
    assert client.get("/auth/github/callback", query_string={
        "state": state, "code": "code",
    }).status_code == 400
    exchange.post.assert_not_called()
    github_auth.CopilotRuntime.assert_not_called()


def test_oauth_refusal_with_valid_state(app, exchange):
    client = app.test_client()
    state = _begin(client)["state"][0]
    assert client.get("/auth/github/callback", query_string={
        "state": state, "error": "access_denied",
    }).status_code == 400
    exchange.post.assert_not_called()


@pytest.mark.parametrize("payload", [
    {}, {"error": "bad_verification_code"},
    {"access_token": "not-oauth", "token_type": "bearer"},
    {"access_token": "gho_value", "token_type": "not-bearer"},
    {"access_token": None}, [], None,
])
def test_invalid_token_response_is_generic(app, exchange, payload):
    exchange.post.return_value.json.return_value = payload
    response = _login(app.test_client())
    assert response.status_code == 502
    assert "bad_verification_code" not in response.text
    exchange.get.assert_not_called()
    github_auth.CopilotRuntime.assert_not_called()


@pytest.mark.parametrize("payload", [{}, {"login": "user"}, {"id": 1}, None, []])
def test_invalid_identity_creates_no_runtime(app, exchange, payload):
    exchange.get.return_value.json.return_value = payload
    assert _login(app.test_client()).status_code == 502
    github_auth.CopilotRuntime.assert_not_called()


def test_network_error_does_not_expose_token(app, exchange):
    exchange.get.side_effect = httpx.RequestError(exchange.token)
    response = _login(app.test_client())
    assert response.status_code == 502
    assert exchange.token not in response.text
    github_auth.CopilotRuntime.assert_not_called()


def test_browser_runtimes_are_isolated_and_logout_requires_csrf(app, exchange):
    first, second = app.test_client(), app.test_client()
    assert _login(first).status_code == 302
    runtime_one = github_auth.CopilotRuntime.return_value
    with first.session_transaction() as cookie:
        key_one = cookie["github_session"]
    runtime_one = github_auth._vault.get(key_one).runtime
    assert _login(second).status_code == 302
    with second.session_transaction() as cookie:
        key_two = cookie["github_session"]
    runtime_two = github_auth._vault.get(key_two).runtime
    assert key_one != key_two
    assert runtime_one is not runtime_two
    with app.test_request_context():
        from flask import session
        session["github_session"] = key_one
        assert github_auth.get_user_runtime() is runtime_one
        session["github_session"] = key_two
        assert github_auth.get_user_runtime() is runtime_two
        session.clear()
        assert github_auth.get_user_runtime() is None
    assert github_auth.get_user_runtime() is None
    assert first.post("/auth/logout").status_code == 400
    assert first.post("/auth/logout", data={"csrf_token": "wrong"}).status_code == 400
    runtime_one.close.assert_not_called()
    csrf = first.get("/identity").json["csrf_token"]
    assert first.post("/auth/logout", data={"csrf_token": csrf}).status_code == 302
    runtime_one.close.assert_called_once()
    assert github_auth._vault.get(key_one) is None
    assert github_auth._vault.get(key_two).runtime is runtime_two
    assert not first.get("/identity").json["logged_in"]
    csrf_two = second.get("/identity").json["csrf_token"]
    assert second.post("/auth/logout", headers={"X-CSRF-Token": csrf_two}).status_code == 302
    runtime_two.close.assert_called_once()
    assert second.get("/auth/logout").status_code == 405


def test_relogin_replaces_and_closes_previous_runtime(app, exchange):
    client = app.test_client()
    _login(client)
    with client.session_transaction() as cookie:
        old_key = cookie["github_session"]
    old_runtime = github_auth._vault.get(old_key).runtime
    _login(client)
    old_runtime.close.assert_called_once()
    assert github_auth._vault.get(old_key) is None
    assert client.get("/identity").json["logged_in"]


def test_vault_expiry_capacity_and_cleanup_outside_lock(app, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(github_auth.time, "monotonic", lambda: clock[0])
    vault = github_auth.TokenVault()

    def runtime():
        result = Mock()

        def close():
            assert vault._lock.acquire(blocking=False)
            vault._lock.release()

        result.close.side_effect = close
        return result

    oldest = runtime()
    old_key = vault.insert(oldest, "old")
    for index in range(32):
        clock[0] += 1
        vault.insert(runtime(), str(index))
    assert len(vault._records) == 32
    assert vault.get(old_key) is None
    oldest.close.assert_called_once()
    vault.close()
    recent = runtime()
    key = vault.insert(recent, "recent")
    clock[0] += 3599
    assert vault.get(key).runtime is recent
    clock[0] += 1
    assert vault.get(key) is None  # Activity cannot extend the absolute lifetime.
    recent.close.assert_called_once()
    expired = runtime()
    vault.insert(expired, "expired")
    clock[0] += 3600
    vault.insert(runtime(), "new")
    expired.close.assert_called_once()
    assert len(vault._records) == 1
    vault.close()


def test_pending_flow_expires_and_is_bounded(app, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(github_auth.time, "monotonic", lambda: clock[0])
    vault = github_auth._vault
    state, _ = vault.begin_oauth()
    clock[0] += vault.OAUTH_TTL
    assert vault.consume_oauth(state, state) is None
    for _ in range(40):
        vault.begin_oauth()
    assert len(vault._pending) == 32
