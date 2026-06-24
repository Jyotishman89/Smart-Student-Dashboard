"""Constants and runtime settings.

Settings (DATABASE_URL, COOKIE_SECRET) are read from Streamlit secrets first,
then environment variables, so the module works both inside a Streamlit run and
in plain scripts/tests where ``st.secrets`` is unavailable.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- academics ---
# Default subjects/components seeded for a brand-new semester. Fully editable
# per user/semester after signup — these are only starting values.
DEFAULT_SUBJECTS: list[tuple[str, float]] = [
    # (subject name, credits)
    ("Data Structures", 4),
    ("DS Lab", 2),
    ("Mathematics III", 3),
    ("Discrete Mathematics", 4),
    ("IT Workshop", 3),
    ("Idea Lab", 0),
    ("Digital Logic Design", 4),
    ("DLD Lab", 1),
]

# (component name, max marks) — must sum to 100 by default.
DEFAULT_COMPONENTS: list[tuple[str, float]] = [
    ("Sessional 1", 10),
    ("Mid Term", 30),
    ("Sessional 2", 10),
    ("End Term", 50),
]

# Grade -> grade point mapping used for SGPA/CGPA (10-point scale, O = 10).
GRADE_POINTS: dict[str, int] = {
    "O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "P": 4, "F": 0,
}

# Grade boundaries expressed as *percentage* of the maximum (so they stay
# correct even when component weightage is customised). Ordered high -> low;
# a grade is awarded when percentage >= its cutoff. O requires a perfect 100%.
GRADE_BANDS: list[tuple[float, str]] = [
    (100, "O"), (90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C"), (40, "P"),
]

PASS_PERCENT = 40.0       # pass mark as a percentage
TARGET_PERCENT = 75.0     # personal target line on charts
ATTENDANCE_REQ = 75.0     # minimum attendance percentage
DEFAULT_PERF_THRESHOLD = 50.0  # subjects below this %-total get flagged

MIN_PASSWORD_LEN = 8

# ---------------------------------------------------------------- settings ----

def _secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit secrets, then env, then default."""
    try:
        import streamlit as st  # imported lazily so non-UI scripts don't need it
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def database_url() -> str:
    """Postgres URL from secrets/env, or a local SQLite file for dev/tests."""
    url = _secret("DATABASE_URL")
    if url:
        # SQLAlchemy expects the postgresql+psycopg2 dialect; normalise the
        # common 'postgres://' prefix that some providers hand out.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    # Dev/offline fallback — file lives next to the repo root.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return f"sqlite:///{os.path.join(base, 'app.db')}"


def cookie_secret() -> str:
    return _secret("COOKIE_SECRET", "insecure-dev-cookie-secret") or "insecure-dev-cookie-secret"
