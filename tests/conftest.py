"""Pytest fixtures — each test gets a fresh temporary SQLite database.

The DB URL is monkeypatched directly rather than set via env var: ``config``
reads Streamlit secrets *before* the environment, so if a real ``secrets.toml``
(with a production DATABASE_URL) is present, an env var would be ignored and the
tests could run against the production database. Patching ``config.database_url``
guarantees the suite is always hermetic.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from ssd import config, db


@pytest.fixture()
def session(monkeypatch):
    """A clean SQLite-backed session for one test, torn down afterwards."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{path}"
    monkeypatch.setattr(config, "database_url", lambda: url)
    db.reset_engine()
    db.init_db()
    factory = db.get_session_factory()
    sess = factory()
    try:
        yield sess
        sess.commit()
    finally:
        sess.close()
        db.reset_engine()
        try:
            os.remove(path)
        except OSError:
            pass
