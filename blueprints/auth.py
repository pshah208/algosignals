"""GitHub OAuth App authorization-code login for browser-owned Copilot runtimes."""

import base64
import hashlib
import secrets
from hmac import compare_digest
from urllib.parse import urlencode, urlsplit

import httpx
from flask import Blueprint, current_app, jsonify, redirect, request, session

from config import settings
from services import github_auth

bp = Blueprint("auth", __name__, url_prefix="/auth")


def _configured():
    key = current_app.secret_key
    uri = getattr(settings, "GITHUB_REDIRECT_URI", "")
    try:
        parsed = urlsplit(uri)
        valid_uri = (
            parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and parsed.path == "/auth/github/callback"
            and (
                parsed.scheme == "https"
                or (
                    parsed.scheme == "http"
                    and settings.FLASK_ENV == "development"
                    and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                )
            )
        )
        parsed.port  # Reject malformed port numbers before redirecting.
    except (TypeError, ValueError):
        valid_uri = False
    return (
        isinstance(key, (str, bytes))
        and len(key) >= 32
        and key not in {
            "dev-secret-key", "change-me-in-production",
            b"dev-secret-key", b"change-me-in-production",
        }
        and all(
            getattr(settings, name, "")
            for name in ("GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "GITHUB_REDIRECT_URI")
        )
        and valid_uri
    )


@bp.get("/github/login")
def login():
    if not _configured():
        return jsonify(error="GitHub login is not configured securely."), 503
    old_state = session.pop("github_oauth_state", None)
    github_auth._vault.consume_oauth(old_state, None)
    state, verifier = github_auth._vault.begin_oauth()
    session["github_oauth_state"] = state
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    response = redirect("https://github.com/login/oauth/authorize?" + urlencode({
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }))
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@bp.get("/github/callback")
def callback():
    verifier = github_auth._vault.consume_oauth(
        session.pop("github_oauth_state", None), request.args.get("state")
    )
    if not _configured():
        return jsonify(error="GitHub login is not configured securely."), 503
    code = request.args.get("code")
    if not verifier or request.args.get("error") or not code:
        return jsonify(error="GitHub login was refused or the callback is invalid."), 400
    runtime = None
    try:
        exchange = httpx.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_REDIRECT_URI,
                "code_verifier": verifier,
            },
            timeout=10,
        )
        exchange.raise_for_status()
        payload = exchange.json()
        token = payload.get("access_token")
        if (
            payload.get("error")
            or not isinstance(token, str)
            or not token.startswith("gho_")
            or len(token) <= 4
            or payload.get("token_type", "").lower() != "bearer"
        ):
            raise ValueError("Invalid OAuth token response")
        identity = httpx.get(
            "https://api.github.com/user",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer " + token,
            },
            timeout=10,
        )
        identity.raise_for_status()
        user = identity.json()
        username = user.get("login")
        if (
            not isinstance(username, str) or not username
            or not isinstance(user.get("id"), int) or user["id"] <= 0
        ):
            raise ValueError("Invalid GitHub identity response")
        runtime = github_auth.CopilotRuntime(token=token)
        key = github_auth._vault.insert(runtime, username)
    except Exception:
        if runtime is not None:
            github_auth._close([runtime])
        return jsonify(error="GitHub login could not be completed. Please try again."), 502
    github_auth._vault.remove(session.get("github_session"))
    session.clear()
    session["github_session"] = key
    session["github_csrf"] = secrets.token_urlsafe(32)
    response = redirect("/")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@bp.post("/logout")
def logout():
    expected = session.get("github_csrf")
    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if (
        not isinstance(expected, str) or not expected
        or not isinstance(submitted, str)
        or not compare_digest(expected.encode(), submitted.encode())
    ):
        return jsonify(error="Invalid logout CSRF token."), 400
    github_auth._vault.remove(session.get("github_session"))
    github_auth._vault.consume_oauth(session.get("github_oauth_state"), None)
    session.clear()
    return redirect("/")
