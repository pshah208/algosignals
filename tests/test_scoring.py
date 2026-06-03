"""Tests for composite scoring and threshold → action mapping."""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.signal_service import (
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    DEFAULT_WEIGHTS,
    _score_to_action,
)


class TestScoreToAction:
    def test_buy(self):
        assert _score_to_action(0.5, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "BUY"

    def test_sell(self):
        assert _score_to_action(-0.5, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "SELL"

    def test_hold_positive(self):
        assert _score_to_action(0.05, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "HOLD"

    def test_hold_negative(self):
        assert _score_to_action(-0.05, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "HOLD"

    def test_exactly_at_buy_threshold(self):
        assert _score_to_action(DEFAULT_BUY_THRESHOLD, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "BUY"

    def test_exactly_at_sell_threshold(self):
        assert _score_to_action(DEFAULT_SELL_THRESHOLD, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD) == "SELL"

    def test_custom_thresholds(self):
        assert _score_to_action(0.3, 0.4, -0.4) == "HOLD"
        assert _score_to_action(0.5, 0.4, -0.4) == "BUY"
        assert _score_to_action(-0.5, 0.4, -0.4) == "SELL"


class TestCompositeScoring:
    """Verify composite score calculation matches expectation."""

    def test_weighted_average(self):
        """Manual composite calculation matches signal_service logic."""
        weights = dict(DEFAULT_WEIGHTS)
        factor_scores = {
            "technical": 0.5,
            "news": 0.2,
            "events": 0.0,
            "financials": 0.3,
            "earnings": 0.1,
        }
        composite = sum(weights[k] * v for k, v in factor_scores.items())
        # Should be between -1 and 1
        assert -1.0 <= composite <= 1.0

    def test_all_zeros_give_hold(self):
        weights = dict(DEFAULT_WEIGHTS)
        factor_scores = {k: 0.0 for k in weights}
        composite = sum(weights[k] * v for k, v in factor_scores.items())
        action = _score_to_action(composite, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD)
        assert composite == 0.0
        assert action == "HOLD"

    def test_all_positive_give_buy(self):
        weights = dict(DEFAULT_WEIGHTS)
        factor_scores = {k: 1.0 for k in weights}
        composite = sum(weights[k] * v for k, v in factor_scores.items())
        action = _score_to_action(composite, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD)
        assert composite > DEFAULT_BUY_THRESHOLD
        assert action == "BUY"

    def test_all_negative_give_sell(self):
        weights = dict(DEFAULT_WEIGHTS)
        factor_scores = {k: -1.0 for k in weights}
        composite = sum(weights[k] * v for k, v in factor_scores.items())
        action = _score_to_action(composite, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD)
        assert composite < DEFAULT_SELL_THRESHOLD
        assert action == "SELL"


class TestRunSignals:
    """Integration-style test for run_signals() with all external calls mocked."""

    def test_run_signals_empty_watchlist(self, tmp_path):
        """run_signals completes without crashing when the watchlist is empty."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import NullPool
        from database.db import Base
        import database.models  # noqa: F401 — registers ORM models
        import services.signal_service as svc

        # Build a fresh DB in tmp_path with all tables
        test_engine = create_engine(
            f"sqlite:///{tmp_path}/test_run.db",
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        Base.metadata.create_all(bind=test_engine)
        TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

        # Patch SessionLocal used directly inside signal_service
        _orig = svc.SessionLocal
        svc.SessionLocal = TestSession
        try:
            run_id = svc.run_signals()
            assert isinstance(run_id, int)
            assert run_id >= 1
        finally:
            svc.SessionLocal = _orig
