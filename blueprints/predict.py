"""Prediction blueprint for LSTM-based price forecasts."""

from flask import Blueprint, jsonify, render_template

from services.prediction.lstm_predictor import predict_next_30d

bp = Blueprint("predict", __name__, url_prefix="/predict")


@bp.route("/<symbol>")
def show(symbol: str):
    payload = predict_next_30d(symbol)
    return render_template("predict.html", symbol=symbol.upper(), payload=payload)


@bp.route("/api/<symbol>")
def api(symbol: str):
    return jsonify(predict_next_30d(symbol))

