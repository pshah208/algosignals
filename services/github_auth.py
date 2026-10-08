"""Single-process browser authentication; credentials never enter Flask cookies."""

import atexit
import secrets
import threading
import time
from dataclasses import dataclass
from hmac import compare_digest

from flask import has_request_context, session

from services.llm.copilot_client import CopilotRuntime


@dataclass
class _Login:
    runtime: CopilotRuntime
    login: str
    created: float
    accessed: float


def _close(runtimes):
    for runtime in runtimes:
        try:
            runtime.close()
        except Exception:
            # Cleanup failures must not expose SDK credentials or block logout.
            pass


class TokenVault:
    """Bounded token-bearing runtimes, with both idle and absolute expiry."""

    MAX_SESSIONS = 32
    TTL = 3600
    OAUTH_TTL = 600

    def __init__(self):
        self._lock = threading.Lock()
        self._records = {}
        self._pending = {}

    def _collect(self, now):
        expired = [
            key for key, record in self._records.items()
            if now - record.created >= self.TTL or now - record.accessed >= self.TTL
        ]
        runtimes = [self._records.pop(key).runtime for key in expired]
        for state, (_, created) in list(self._pending.items()):
            if now - created >= self.OAUTH_TTL:
                self._pending.pop(state)
        return runtimes

    def begin_oauth(self):
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        with self._lock:
            now = time.monotonic()
            expired = self._collect(now)
            if len(self._pending) >= self.MAX_SESSIONS:
                self._pending.pop(next(iter(self._pending)))
            self._pending[state] = (verifier, now)
        _close(expired)
        return state, verifier

    def consume_oauth(self, expected, received):
        with self._lock:
            expired = self._collect(time.monotonic())
            pending = self._pending.pop(expected, None) if isinstance(expected, str) else None
        _close(expired)
        if (
            not pending
            or not isinstance(received, str)
            or not compare_digest(expected.encode(), received.encode())
        ):
            return None
        return pending[0]

    def insert(self, runtime, login):
        key = secrets.token_urlsafe(32)
        with self._lock:
            now = time.monotonic()
            expired = self._collect(now)
            if len(self._records) >= self.MAX_SESSIONS:
                oldest = min(self._records, key=lambda k: self._records[k].accessed)
                expired.append(self._records.pop(oldest).runtime)
            self._records[key] = _Login(runtime, login, now, now)
        _close(expired)
        return key

    def get(self, key):
        with self._lock:
            now = time.monotonic()
            expired = self._collect(now)
            record = self._records.get(key) if isinstance(key, str) else None
            if record:
                record.accessed = now
        _close(expired)
        return record

    def remove(self, key):
        with self._lock:
            record = self._records.pop(key, None) if isinstance(key, str) else None
        _close([record.runtime] if record else [])

    def close(self):
        with self._lock:
            runtimes = [record.runtime for record in self._records.values()]
            self._records.clear()
            self._pending.clear()
        _close(runtimes)


_vault = TokenVault()
atexit.register(_vault.close)


def get_user_runtime():
    """Return only the current browser's runtime; never fall back to host auth."""
    if not has_request_context():
        return None
    record = _vault.get(session.get("github_session"))
    return record.runtime if record else None


def get_login_info():
    """Expose identity and a logout CSRF token, but no GitHub credential."""
    if not has_request_context():
        return {"logged_in": False, "login": None, "csrf_token": None}
    record = _vault.get(session.get("github_session"))
    if not record:
        session.pop("github_session", None)
    csrf = session.get("github_csrf")
    if not isinstance(csrf, str) or not csrf:
        csrf = secrets.token_urlsafe(32)
        session["github_csrf"] = csrf
    return {
        "logged_in": record is not None,
        "login": record.login if record else None,
        "csrf_token": csrf,
    }
