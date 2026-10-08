"""Offline integration tests for bounded signal processing and price freshness."""

import importlib
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from database.db import Base
from database.models import AppConfig, Recommendation, SignalRun, Watchlist
from services import signal_service as svc
from services.factors import FactorResult


@pytest.fixture()
def database(monkeypatch):
    owner = threading.get_ident()
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    class CallingThreadSession(Session):
        def __getattribute__(self, name):
            if name in {"query", "add", "commit", "refresh", "rollback", "close"}:
                assert threading.get_ident() == owner
            return super().__getattribute__(name)

    @event.listens_for(engine, "before_cursor_execute")
    def check_sql_thread(*args):
        assert threading.get_ident() == owner

    original_getattribute = Watchlist.__getattribute__

    def check_watchlist_thread(self, name):
        if name in {"symbol", "exchange", "enabled"}:
            assert threading.get_ident() == owner
        return original_getattribute(self, name)

    monkeypatch.setattr(Watchlist, "__getattribute__", check_watchlist_thread)
    factory = sessionmaker(bind=engine, class_=CallingThreadSession, autoflush=False)
    monkeypatch.setattr(svc, "SessionLocal", factory)
    yield factory
    engine.dispose()


def seed(database, symbols):
    with database() as db:
        db.add_all(
            Watchlist(symbol=symbol, exchange=f"EX-{symbol}") for symbol in symbols
        )
        db.add(Watchlist(symbol="DISABLED", enabled=False))
        weights = {name: float(name == "technical") for name in svc.DEFAULT_WEIGHTS}
        db.add_all([
            AppConfig(key="factor_weights", value=json.dumps(weights)),
            AppConfig(key="buy_threshold", value="0.4"),
            AppConfig(key="sell_threshold", value="-0.4"),
        ])
        db.commit()


@pytest.mark.parametrize("inject_client", [True, False])
def test_multi_symbol_pipeline_bounded_and_ordered(database, monkeypatch, inject_client):
    symbols = [f"S{i}" for i in range(8)]
    seed(database, symbols)
    owner = threading.get_ident()
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = 0
    calls = {symbol: [] for symbol in symbols}
    worker_threads = set()
    scores = [0.45006, -0.45006, 0.01, 0.35, 0.5, -0.5, 0.0, 0.4]

    class Client:
        enabled = True
        model = "request-specific"

        def __bool__(self):
            return False

        def generate_rationale(self, **kwargs):
            symbol = kwargs["symbol"]
            assert threading.get_ident() != owner
            assert calls[symbol] == list(svc.DEFAULT_WEIGHTS)
            assert kwargs["factor_scores"]["technical"] == round(
                scores[int(symbol[1:])], 4
            )
            if symbol == "S1":
                raise RuntimeError("LLM unavailable")
            return f"{self.model}:{symbol}:{kwargs['action']}"

        def _fallback_rationale(self, symbol, factor_scores, composite, action):
            return f"fallback:{symbol}:{action}"

    client = Client()
    default_client = Mock(return_value=client)
    monkeypatch.setattr(svc, "get_llm_client", default_client)

    def compute(name):
        def factor(symbol, llm_client=None):
            nonlocal active, peak
            assert threading.get_ident() != owner
            if name == "news":
                assert llm_client is client
            with lock:
                worker_threads.add(threading.get_ident())
                calls[symbol].append(name)
                if name == "technical":
                    active += 1
                    peak = max(peak, active)
                elif name == "earnings":
                    active -= 1
            if name == "technical":
                barrier.wait(timeout=10)
            if name == "events" and symbol == "S3":
                raise RuntimeError("provider unavailable")
            return FactorResult(
                score=scores[int(symbol[1:])],
                rationale=f"{name}:{symbol}",
                source="offline",
                raw_values={"symbol": symbol},
            )
        return factor

    for name in svc.DEFAULT_WEIGHTS:
        module = importlib.import_module(f"services.factors.{name}_signals")
        monkeypatch.setattr(module, "compute", compute(name))

    run_id = svc.run_signals(client) if inject_client else svc.run_signals()
    assert active == 0
    assert peak == 4
    assert len(worker_threads) == 4
    assert client.model == "request-specific"
    assert default_client.call_count == (0 if inject_client else 1)
    with database() as db:
        run = db.get(SignalRun, run_id)
        recs = db.query(Recommendation).order_by(Recommendation.id).all()
        assert run.status == "ok"
        assert [rec.symbol for rec in recs] == symbols
        assert [rec.action for rec in recs] == [
            "BUY", "SELL", "HOLD", "HOLD", "BUY", "SELL", "HOLD", "BUY"
        ]
        assert run.summary == ", ".join(
            f"{rec.symbol}:{rec.action}({rec.composite_score:+.2f})" for rec in recs
        )
        for rec in recs:
            assert rec.run_id == run_id
            assert rec.exchange == f"EX-{rec.symbol}"
            assert rec.composite_score == round(scores[int(rec.symbol[1:])], 4)
            assert rec.llm_used is True
            details = rec.factor_scores_dict
            assert list(details) == list(svc.DEFAULT_WEIGHTS)
            assert details["technical"]["score"] == scores[int(rec.symbol[1:])]
            assert details["technical"]["source"] == "offline"
            assert details["technical"]["raw_values"] == {"symbol": rec.symbol}
            if rec.symbol == "S1":
                assert rec.rationale == "fallback:S1:SELL"
            else:
                assert rec.rationale == f"request-specific:{rec.symbol}:{rec.action}"
        assert recs[3].factor_scores_dict["events"]["rationale"] == "data unavailable"


def test_failed_commit_rolls_back_before_recording_error(database, monkeypatch):
    seed(database, ["S0"])
    session = database()
    original_commit = session.commit
    commits = 0
    rollback = Mock(wraps=session.rollback)
    monkeypatch.setattr(session, "rollback", rollback)

    def commit():
        nonlocal commits
        commits += 1
        if commits == 2:
            for rec in list(session.new):
                if isinstance(rec, Recommendation):
                    rec.id = 1
            session.add(Recommendation(id=1, run_id=1, symbol="DUPLICATE"))
        if commits == 3:
            assert rollback.call_count == 1
            assert session.is_active
        original_commit()

    monkeypatch.setattr(session, "commit", commit)
    monkeypatch.setattr(svc, "SessionLocal", lambda: session)
    monkeypatch.setattr(svc, "_run_factor", lambda *args: FactorResult())
    client = SimpleNamespace(
        enabled=False, generate_rationale=lambda **kwargs: "offline"
    )
    run_id = svc.run_signals(client)
    assert commits == 3
    assert rollback.call_count == 1
    with database() as db:
        run = db.get(SignalRun, run_id)
        assert run.status == "error"
        assert run.summary == "Run failed — see logs."
        assert db.query(Recommendation).count() == 0


@pytest.fixture()
def price_provider(monkeypatch):
    svc.get_current_price.cache_clear()
    ticker = Mock()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker))
    yield ticker
    svc.get_current_price.cache_clear()


def test_price_cache_expires_using_monotonic_clock(price_provider, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(svc.time, "monotonic", lambda: clock[0])
    price_provider.side_effect = [
        SimpleNamespace(fast_info={"last_price": 123.456789, "currency": "USD"}),
        SimpleNamespace(fast_info={"last_price": 124.0, "currency": "USD"}),
    ]
    first = svc.get_current_price("S0")
    assert first == {"price": 123.4568, "currency": "USD"}
    first["price"] = -1
    clock[0] += svc.PRICE_CACHE_TTL_SECONDS - 0.01
    assert svc.get_current_price("S0")["price"] == 123.4568
    assert price_provider.call_count == 1
    clock[0] = 100.0 + svc.PRICE_CACHE_TTL_SECONDS
    assert svc.get_current_price("S0")["price"] == 124.0
    assert price_provider.call_count == 2


@pytest.mark.parametrize("failure", [
    RuntimeError("network failed"),
    SimpleNamespace(fast_info={}, info={"currency": "USD"}),
])
def test_price_failures_retry_immediately(price_provider, failure):
    price_provider.side_effect = [
        failure, SimpleNamespace(fast_info={"lastPrice": 42, "currency": "USD"})
    ]
    assert svc.get_current_price("S0")["price"] is None
    assert svc.get_current_price("S0") == {"price": 42.0, "currency": "USD"}
    assert price_provider.call_count == 2


def test_price_cache_is_bounded_and_clearable(price_provider):
    price_provider.return_value = SimpleNamespace(
        fast_info={"last_price": 10, "currency": "USD"}
    )
    for i in range(svc.PRICE_CACHE_MAXSIZE + 1):
        svc.get_current_price(f"S{i}")
    assert len(svc._price_cache) == svc.PRICE_CACHE_MAXSIZE
    svc.get_current_price("S0")
    assert price_provider.call_count == svc.PRICE_CACHE_MAXSIZE + 2
    svc.get_current_price.cache_clear()
    assert not svc._price_cache
    svc.get_current_price("S0")
    assert price_provider.call_count == svc.PRICE_CACHE_MAXSIZE + 3
