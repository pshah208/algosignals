"""Model-selection API blueprint — /api/models.

GET  /api/models  → list available models and the currently active one.
POST /api/models  → set the active model (validated against the allowed list).
"""

from flask import Blueprint, flash, jsonify, redirect, request, session, url_for

from config import settings

bp = Blueprint("models", __name__, url_prefix="/api/models")

_SESSION_KEY = "active_llm_model"


def get_active_model() -> str:
    """Return the model currently selected for this session (or the default)."""
    default = settings.COPILOT_MODEL if settings.LLM_PROVIDER == "copilot" else settings.LLM_MODEL
    return session.get(f"{_SESSION_KEY}:{settings.LLM_PROVIDER}", default)


def available_models() -> list[str]:
    if settings.LLM_PROVIDER == "copilot":
        from services.llm.copilot_client import get_request_copilot_runtime

        try:
            return get_request_copilot_runtime().metadata()["available"] or [settings.COPILOT_MODEL]
        except RuntimeError:
            return [settings.COPILOT_MODEL]
    return settings.llm_available_models


@bp.route("", methods=["GET"])
def list_models():
    """Return available models and the currently active one.

    Response JSON::

        {
            "available": ["openai/gpt-4.1-mini", ...],
            "active": "openai/gpt-4.1-mini"
        }
    """
    return jsonify(
        {
            "available": available_models(),
            "active": get_active_model(),
        }
    )


@bp.route("", methods=["POST"])
def set_model():
    """Set the active LLM model for the current session.

    Accepts either ``application/json`` with ``{"model": "..."}`` or a plain
    HTML form POST with a ``model`` field.

    Returns HTTP 400 when the requested model is not in the allowed list.
    For HTML form POSTs, redirects back to the dashboard with a flash message.
    """
    is_json = request.is_json
    if is_json:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        model = data.get("model", "")
    else:
        model = request.form.get("model", "")

    if not isinstance(model, str):
        return jsonify({"error": "model must be a string"}), 400
    model = model.strip()
    allowed = available_models()

    if not model:
        if is_json:
            return jsonify({"error": "model field is required"}), 400
        flash("No model selected.", "warning")
        return redirect(url_for("dashboard.index"))

    if model not in allowed:
        if is_json:
            return jsonify(
                {"error": f"Unknown model '{model}'. Allowed: {allowed}"}
            ), 400
        flash(f"Unknown model '{model}'.", "danger")
        return redirect(url_for("dashboard.index"))

    session[f"{_SESSION_KEY}:{settings.LLM_PROVIDER}"] = model

    if is_json:
        return jsonify({"active": model})

    flash(f"LLM model switched to {model}.", "success")
    return redirect(url_for("dashboard.index"))
