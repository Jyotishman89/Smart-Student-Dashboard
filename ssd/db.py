"""Database engine and session management.

The engine is created once and cached. Inside a Streamlit run it is cached with
``st.cache_resource`` (one engine per server process, connection-pooled). Outside
Streamlit (scripts, tests) it falls back to a module-level singleton.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from . import config
from .models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _build_engine(url: str) -> Engine:
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # Allow use across Streamlit's threads for the dev SQLite fallback.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Neon-friendly pooling: keep the pool small and recycle stale conns.
        kwargs.update(pool_size=5, max_overflow=5, pool_recycle=300)
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    """Return a process-wide cached engine (Streamlit-aware)."""
    global _engine
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _cached_engine(url: str) -> Engine:
            return _build_engine(url)

        return _cached_engine(config.database_url())
    except Exception:
        # Not running under Streamlit (script/test) — use a plain singleton.
        if _engine is None:
            _engine = _build_engine(config.database_url())
        return _engine


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context. Commits on success, rolls back on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables if they do not yet exist (idempotent)."""
    Base.metadata.create_all(get_engine())


def reset_engine() -> None:
    """Drop cached engine/session factory (used by tests to swap databases)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
