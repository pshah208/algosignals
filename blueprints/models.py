"""Model-selection API blueprint — /api/models.

GET  /api/models  → list available models and the currently active one.
POST /api/models  → set the active model (validated against the allowed list).
"""

from flask import Blueprint, flash, jsonify, redirect, request, session, url_for

from config import settings
from services.llm.llm_client import get_llm_client

bp = Blueprint("models", __name__, url_prefix="/api/models")

_SESSION_KEY = "active_llm_model"


def get_active_model() -> str:
    """Return the model currently selected for this session (or the default)."""
    return session.get(_SESSION_KEY, settings.LLM_MODEL)


def apply_session_model() -> None:
    """Sync the session-stored model selection onto the shared LLM client."""
    model = get_active_model()
    get_llm_client().set_model(model)


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
            "available": settings.llm_available_models,
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
        model = data.get("model", "")
    else:
        model = request.form.get("model", "")

    model = model.strip()
    allowed = settings.llm_available_models

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

    session[_SESSION_KEY] = model
    get_llm_client().set_model(model)

    if is_json:
        return jsonify({"active": model})

    flash(f"LLM model switched to {model}.", "success")
    return redirect(url_for("dashboard.index"))
