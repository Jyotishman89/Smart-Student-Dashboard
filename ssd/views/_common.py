"""Helpers shared across page views."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .. import academics, auth
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


# --------------------------------------------------- snapshot browse mode ----
# When a snapshot is being "viewed", every page renders that snapshot's data
# (read-only) instead of the live database state. The choice lives in session
# state (per browser session) and is set from the History page; it never writes
# to the database, so the live data and manual SGPA/CGPA settings stay untouched.
_VIEW_KEY = "view_snapshot_id"


def viewing_snapshot_id() -> int | None:
    return st.session_state.get(_VIEW_KEY)


def set_viewing_snapshot(snapshot_id: int | None) -> None:
    if snapshot_id is None:
        st.session_state.pop(_VIEW_KEY, None)
    else:
        st.session_state[_VIEW_KEY] = int(snapshot_id)


def _snapshot_to_state(snap: dict, semester_id: int) -> dict[str, Any]:
    """Rebuild a ``get_state``-shaped dict from a snapshot's payload so the same
    rendering code can show a snapshot read-only. The snapshot's recorded SGPA is
    placed as the semester override so the cards show that exact value."""
    payload = snap.get("payload") or {}
    comps = payload.get("components", [])
    return {
        "semester": {
            "id": semester_id,
            "label": payload.get("semester_label", ""),
            "perf_threshold": payload.get("perf_threshold", 50.0),
            "is_active": True,
            "sgpa_override": snap.get("sgpa"),
        },
        "components": [
            {"id": -(i + 1), "name": cc["name"], "max_marks": cc["max_marks"],
             "order_index": cc.get("order_index", i)}
            for i, cc in enumerate(comps)
        ],
        "subjects": [
            {"id": -(i + 1), "name": s["name"], "credits": s["credits"],
             "order_index": s.get("order_index", i),
             "scores": {cc["name"]: float(s.get("scores", {}).get(cc["name"], 0.0))
                        for cc in comps},
             "attendance": {
                 "held": int(s.get("attendance", {}).get("held", 0)),
                 "attended": int(s.get("attendance", {}).get("attended", 0))}}
            for i, s in enumerate(payload.get("subjects", []))
        ],
    }


def page_state(user_id: int, semester_id: int) -> tuple[dict[str, Any], dict | None]:
    """Return ``(state, viewing)`` for a page.

    ``viewing`` is the snapshot dict currently being browsed (or ``None`` for
    live data). When browsing, ``state`` is rebuilt read-only from that snapshot
    so every page reflects it; otherwise it is the live database state.
    """
    snap_id = viewing_snapshot_id()
    if snap_id is not None:
        with session_scope() as session:
            snap = repo.get_snapshot(session, snap_id)
            if (snap is not None and snap.user_id == user_id
                    and snap.semester_id == semester_id):
                snap_dict = {
                    "id": snap.id, "taken_at": snap.taken_at, "sgpa": snap.sgpa,
                    "total_credits": snap.total_credits, "payload": snap.payload or {},
                }
                return _snapshot_to_state(snap_dict, semester_id), snap_dict
        set_viewing_snapshot(None)  # stale (deleted / other semester) → fall to live
    return load_state(semester_id), None


def view_banner(viewing: dict | None) -> None:
    """Top-of-page banner shown while browsing a snapshot, with a return button."""
    if not viewing:
        return
    left, right = st.columns([4, 1])
    left.info(f"👁️ Viewing the snapshot saved **{viewing['taken_at']:%Y-%m-%d %H:%M}** "
              f"(SGPA {academics.round_2dp_from_float(viewing['sgpa'])}) — read-only. "
              "Your live data and manual SGPA/CGPA settings are unchanged.")
    if right.button("↩️ Return to live", use_container_width=True, key="btn_return_live"):
        set_viewing_snapshot(None)
        st.rerun()


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
