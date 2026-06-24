"""Pytest fixtures — each test gets a fresh temporary SQLite database."""
from __future__ import annotations

import os
import tempfile

import pytest

from ssd import db


@pytest.fixture()
def session():
    """A clean DB-backed session for one test, torn down afterwards."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
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
        os.environ.pop("DATABASE_URL", None)
        try:
            os.remove(path)
        except OSError:
            pass
