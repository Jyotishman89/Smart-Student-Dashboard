"""Helpers shared across page views."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .. import auth
from .. import repository as repo
from ..db import session_scope


def require_user() -> int:
    uid = auth.current_user_id()
    if not uid:
        st.warning("Please log in to continue.")
        st.stop()
    return uid


def current_semester_id(user_id: int) -> int:
    """Resolve the active semester id, honouring the sidebar selection.

    The sidebar runs before every page and stores a validated
    ``active_semester_id`` in session state (it self-heals if the stored id is
    gone), so trust it when present and skip a redundant DB round-trip. Only
    query on the rare path where it is unset.
    """
    sid = st.session_state.get("active_semester_id")
    if sid is not None:
        return sid
    with session_scope() as session:
        active = repo.get_active_semester(session, user_id)
        st.session_state.active_semester_id = active.id
        return active.id


def load_state(semester_id: int) -> dict[str, Any]:
    with session_scope() as session:
        return repo.get_state(session, semester_id)


# ----------------------------------------------------- dataframe converters ---
def marks_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    comp_names = [c["name"] for c in state["components"]]
    data = {"Subject": [s["name"] for s in state["subjects"]]}
    for cn in comp_names:
        data[cn] = [s["scores"].get(cn, 0.0) for s in state["subjects"]]
    return pd.DataFrame(data)


def attendance_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "Subject": [s["name"] for s in state["subjects"]],
        "Classes Held": [s["attendance"]["held"] for s in state["subjects"]],
        "Classes Attended": [s["attendance"]["attended"] for s in state["subjects"]],
    })


def subjects_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "id": [s["id"] for s in state["subjects"]],
        "Subject": [s["name"] for s in state["subjects"]],
        "Credits": [s["credits"] for s in state["subjects"]],
    })


def components_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "id": [c["id"] for c in state["components"]],
        "Component": [c["name"] for c in state["components"]],
        "Max Marks": [c["max_marks"] for c in state["components"]],
    })


def chart_type(key: str) -> str:
    return st.selectbox("Chart type", ["Bar", "Pie", "Line"], key=key)
