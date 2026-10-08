"""Tests for runtime LLM model selection.

Covers:
- Model list comes from config (including LLM_MODEL always first).
- Default model is selected initially.
- Switching to a valid model updates the LLMClient.
- Switching to an invalid model is rejected (400 / flash + redirect).
- No-token (llm_enabled=False) path degrades gracefully.
"""

import json
from unittest.mock import patch

import pytest

from config import Config
from services.llm.llm_client import LLMClient, get_llm_client


# ---------------------------------------------------------------------------
# Config unit tests
# ---------------------------------------------------------------------------


class TestLLMAvailableModelsConfig:
    """Config.llm_available_models must include LLM_MODEL as the first entry."""

    def test_default_list_contains_llm_model(self):
        cfg = Config()
        models = cfg.llm_available_models
        assert cfg.LLM_MODEL in models

    def test_llm_model_is_first(self):
        cfg = Config()
        assert cfg.llm_available_models[0] == cfg.LLM_MODEL

    def test_custom_list_without_llm_model(self):
        """When LLM_MODEL is absent from the env list it is prepended."""
        cfg = Config()
        original_model = cfg.LLM_MODEL
        with patch.object(Config, "LLM_AVAILABLE_MODELS", "openai/gpt-4o,openai/gpt-4.1"):
            with patch.object(Config, "LLM_MODEL", "openai/gpt-4.1-mini"):
                models = cfg.llm_available_models
        # LLM_MODEL should be first regardless
        assert models[0] == cfg.LLM_MODEL

    def test_custom_list_with_llm_model_present(self):
        """When LLM_MODEL is already in the list it is moved to first position."""
        cfg = Config()
        with patch.object(
            Config,
            "LLM_AVAILABLE_MODELS",
            "openai/gpt-4o,openai/gpt-4.1-mini,openai/gpt-4.1",
        ):
            with patch.object(Config, "LLM_MODEL", "openai/gpt-4.1-mini"):
                models = cfg.llm_available_models
        assert models[0] == "openai/gpt-4.1-mini"
        assert "openai/gpt-4o" in models

    def test_no_duplicates_when_llm_model_in_list(self):
        cfg = Config()
        with patch.object(
            Config,
            "LLM_AVAILABLE_MODELS",
            f"{cfg.LLM_MODEL},openai/gpt-4o",
        ):
            models = cfg.llm_available_models
        assert models.count(cfg.LLM_MODEL) == 1

    def test_default_list_includes_anthropic_and_microsoft(self):
        cfg = Config()
        models = cfg.llm_available_models
        assert any(m.startswith("anthropic/") for m in models)
        assert any(m.startswith("microsoft/") for m in models)


# ---------------------------------------------------------------------------
# LLMClient.set_model unit tests
# ---------------------------------------------------------------------------


class TestLLMClientSetModel:
    """set_model() updates the active model used for _chat."""

    def test_set_model_changes_model_attribute(self):
        client = LLMClient(token="fake")
        assert client.model != "openai/gpt-4o"
        client.set_model("openai/gpt-4o")
        assert client.model == "openai/gpt-4o"

    def test_set_model_used_in_chat_payload(self):
        client = LLMClient(token="fake-token")
        client.set_model("openai/gpt-4o")

        captured = {}

        import httpx

        def fake_post(url, **kwargs):
            captured.update(kwargs.get("json", {}))
            raise httpx.ConnectError("mocked")

        with patch("httpx.post", side_effect=fake_post):
            client._chat([{"role": "user", "content": "hi"}])

        assert captured.get("model") == "openai/gpt-4o"

    def test_set_model_on_disabled_client(self):
        """set_model works even when the client has no token."""
        client = LLMClient(token="")
        client.set_model("openai/gpt-4o")
        assert client.model == "openai/gpt-4o"
        # Should still degrade gracefully
        assert client._chat([{"role": "user", "content": "hi"}]) is None


# ---------------------------------------------------------------------------
# Flask API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """Create a minimal Flask test app with the models blueprint."""
    from flask import Flask
    from blueprints.models import bp as models_bp
    from blueprints.dashboard import bp as dashboard_bp

    flask_app = Flask(__name__, template_folder="../templates")
    flask_app.secret_key = "test-secret"
    flask_app.register_blueprint(models_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


class TestModelsAPIGet:
    def test_get_returns_available_and_active(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "available" in data
        assert "active" in data
        assert isinstance(data["available"], list)
        assert len(data["available"]) > 0

    def test_default_active_model_is_llm_model(self, client):
        from config import settings

        resp = client.get("/api/models")
        data = resp.get_json()
        assert data["active"] == settings.LLM_MODEL


class TestModelsAPIPost:
    def test_valid_model_json(self, client):
        from config import settings

        target = settings.llm_available_models[0]
        resp = client.post(
            "/api/models",
            data=json.dumps({"model": target}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] == target

    def test_valid_model_updates_llm_client(self, client, app):
        from config import settings
        from services.llm import llm_client as llm_mod

        # Reset singleton
        llm_mod._client = None
        target = settings.llm_available_models[0]
        with app.test_request_context():
            with client as c:
                c.post(
                    "/api/models",
                    data=json.dumps({"model": target}),
                    content_type="application/json",
                )
        assert get_llm_client().model == target

    def test_invalid_model_returns_400(self, client):
        resp = client.post(
            "/api/models",
            data=json.dumps({"model": "invalid/unknown-model"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_empty_model_returns_400(self, client):
        resp = client.post(
            "/api/models",
            data=json.dumps({"model": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_missing_model_field_returns_400(self, client):
        resp = client.post(
            "/api/models",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_form_post_valid_model_redirects(self, client):
        from config import settings

        target = settings.llm_available_models[0]
        resp = client.post("/api/models", data={"model": target})
        # HTML form posts should redirect (302)
        assert resp.status_code == 302

    def test_form_post_invalid_model_redirects(self, client):
        resp = client.post("/api/models", data={"model": "bad/model"})
        assert resp.status_code == 302

    def test_session_persists_model_selection(self, client):
        from config import settings

        models = settings.llm_available_models
        if len(models) < 2:
            pytest.skip("Need at least two models to test switching")

        target = models[1]  # pick the second model

        with client.session_transaction() as sess:
            sess.clear()

        client.post(
            "/api/models",
            data=json.dumps({"model": target}),
            content_type="application/json",
        )
        resp = client.get("/api/models")
        data = resp.get_json()
        assert data["active"] == target

    def test_model_switch_does_not_mutate_shared_client(self, client):
        from config import settings
        from services.llm import llm_client as llm_mod

        original = LLMClient(model=settings.LLM_MODEL)
        with patch.object(llm_mod, "_client", original):
            target = settings.llm_available_models[1]
            assert client.post("/api/models", json={"model": target}).status_code == 200
            assert original.model == settings.LLM_MODEL
            assert client.get("/api/models").json["active"] == target

    @pytest.mark.parametrize("data", [["invalid"], {"model": 123}, {"model": {}}])
    def test_invalid_model_types_return_400(self, client, data):
        assert client.post("/api/models", json=data).status_code == 400


# ---------------------------------------------------------------------------
# No-token (llm_enabled=False) graceful degradation
# ---------------------------------------------------------------------------


class TestModelSelectionWithNoToken:
    """Model selection must not crash when LLM is disabled."""

    def test_list_models_works_without_token(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_set_model_works_without_token(self, client):
        from config import settings

        target = settings.llm_available_models[0]
        resp = client.post(
            "/api/models",
            data=json.dumps({"model": target}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_llm_client_still_degrades_after_model_switch(self):
        """After switching model, disabled client must still return fallbacks."""
        client = LLMClient(token="")
        client.set_model("openai/gpt-4o")
        assert client.enabled is False
        result = client.score_news_sentiment("AAPL", ["Some headline"])
        assert result["score"] == 0.0
