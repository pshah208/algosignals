"""API-key protected, bounded Copilot research API."""

import hmac

from flask import Blueprint, jsonify, request

from config import settings
from services.research_service import validate_symbol

bp = Blueprint("research", __name__, url_prefix="/api/v1")


def require_copilot_key(form=False):
    if not settings.APP_API_KEY or settings.APP_API_KEY == "change-me-api-key":
        return jsonify({"error": "Configure APP_API_KEY before using Copilot from the app."}), 503
    value = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if form and not value:
        value = request.form.get("api_key", "")
    if not hmac.compare_digest(value.encode(), settings.APP_API_KEY.encode()):
        return jsonify({"error": "Unauthorized"}), 401
    return None


@bp.route("/copilot/status")
def copilot_status():
    denied = require_copilot_key()
    if denied:
        return denied
    if settings.LLM_PROVIDER != "copilot":
        return jsonify({"enabled": False, "authenticated": False, "available": []})
    from services.llm.copilot_client import get_request_copilot_runtime

    try:
        runtime = get_request_copilot_runtime()
    except RuntimeError:
        return jsonify({"error": "Sign in with GitHub first."}), 401
    return jsonify({"enabled": True, **runtime.metadata()})


@bp.route("/research", methods=["POST"])
def research():
    denied = require_copilot_key()
    if denied:
        return denied
    if settings.LLM_PROVIDER != "copilot":
        return jsonify({"error": "Set LLM_PROVIDER=copilot and sign in with the Copilot CLI."}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    try:
        symbol = validate_symbol(data.get("symbol"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    question = data.get("question", "Explain the evidence, risks and missing data for this ticker.")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        return jsonify({"error": "question must contain 1–2000 characters."}), 400

    from blueprints.models import available_models, get_active_model
    from services.llm.copilot_client import get_request_copilot_runtime

    try:
        runtime = get_request_copilot_runtime()
    except RuntimeError:
        return jsonify({"error": "Sign in with GitHub first."}), 401
    model = get_active_model()
    if model not in available_models():
        return jsonify({"error": "Select an available Copilot model first."}), 400
    try:
        result = runtime.research(symbol, question.strip(), model)
    except Exception:
        return jsonify({"error": "Copilot unavailable or busy. Check CLI login and retry."}), 503
    return jsonify({
        "symbol": symbol,
        **result,
        "disclaimer": "Research only — not investment advice. No orders are placed.",
    })
