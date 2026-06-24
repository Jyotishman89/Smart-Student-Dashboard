"""Database engine and session management.

The engine is created once and cached. Inside a Streamlit run it is cached with
``st.cache_resource`` (one engine per server process, connection-pooled). Outside
Streamlit (scripts, tests) it falls back to a module-level singleton.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
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


# Columns added to existing models after the first release. ``create_all`` only
# creates *missing tables* — it never alters an existing one — so deployed
# databases (Neon, a dev app.db) need these added explicitly.
_ADDITIVE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "semesters": [("sgpa_override", "FLOAT")],
    "users": [("cgpa_override", "FLOAT")],
}


def ensure_schema(engine: Engine | None = None) -> None:
    """Add any missing additive columns via ``ALTER TABLE`` (idempotent).

    Lets a running database self-migrate on startup: brand-new tables already
    have the columns (from ``create_all``); older ones get them added here.
    """
    engine = engine or get_engine()
    inspector = inspect(engine)
    for table, columns in _ADDITIVE_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, sqltype in columns:
            if name in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}"))
            except Exception:
                pass  # raced with another worker that already added it — fine


def init_db() -> None:
    """Create missing tables and add any new columns (idempotent)."""
    Base.metadata.create_all(get_engine())
    ensure_schema()


def reset_engine() -> None:
    """Drop cached engine/session factory (used by tests to swap databases)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
