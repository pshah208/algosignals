"""SQLAlchemy ORM models for AlgoSignals."""

import datetime
import json

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base


class Watchlist(Base):
    """Symbols the user wants to track."""

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "enabled": self.enabled,
        }


class AppConfig(Base):
    """Key/value configuration store."""

    __tablename__ = "config"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")


class SignalRun(Base):
    """One execution of the daily signal pipeline."""

    __tablename__ = "signal_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/ok/error
    summary: Mapped[str] = mapped_column(Text, default="")

    recommendations: Mapped[list["Recommendation"]] = relationship(
        "Recommendation", back_populates="run", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "status": self.status,
            "summary": self.summary,
        }


class Recommendation(Base):
    """A BUY / SELL / HOLD recommendation produced by one run for one symbol."""

    __tablename__ = "recommendation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("signal_run.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), default="")
    action: Mapped[str] = mapped_column(String(10), default="HOLD")  # BUY/SELL/HOLD
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    factor_scores: Mapped[str] = mapped_column(Text, default="{}")  # JSON blob
    rationale: Mapped[str] = mapped_column(Text, default="")
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    run: Mapped["SignalRun"] = relationship("SignalRun", back_populates="recommendations")

    @property
    def factor_scores_dict(self) -> dict:
        try:
            return json.loads(self.factor_scores)
        except (ValueError, TypeError):
            return {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "action": self.action,
            "composite_score": self.composite_score,
            "factor_scores": self.factor_scores_dict,
            "rationale": self.rationale,
            "llm_used": self.llm_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
