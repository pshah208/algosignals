"""SQLAlchemy engine + session factory for AlgoSignals."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

from config import settings
from utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


# Use NullPool so each request creates/closes connections independently (SQLite-safe).
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables that do not yet exist."""
    # Import models so SQLAlchemy registers them before create_all.
    import database.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")


def get_db():
    """Yield a database session and close it after use.

    Designed to be used as a Flask ``g``-scoped dependency.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
