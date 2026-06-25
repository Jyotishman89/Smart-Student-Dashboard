"""Home / overview page."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .. import academics
from .. import repository as repo
from ..db import session_scope
from ..theme import hero
from . import _common as c


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state, viewing = c.page_state(uid, sid)
    summary = repo.summarize_state(state)

    name = st.session_state.get("user_name") or "there"
    hero(
        f"Welcome back, {name.split()[0] if name else 'there'} 👋",
        f"Active semester: {state['semester']['label']}",
        chips=["Marks", "Attendance", "SGPA / CGPA", "Snapshots"],
    )
    c.view_banner(viewing)
    st.write("")

    with session_scope() as session:
        cgpa_val, cgpa_credits, cgpa_manual = repo.cgpa_for_user(session, uid)

    percents = [r["Percent"] for r in summary["rows"]]
    att_vals = [academics.att_percent(s["attendance"]["held"], s["attendance"]["attended"])
                for s in state["subjects"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current SGPA", f"{academics.round_2dp(summary['sgpa_effective'])}",
              help=(f"From snapshot {viewing['taken_at']:%Y-%m-%d %H:%M}." if viewing
                    else ("Set manually in Settings." if summary["sgpa_is_manual"] else None)))
    k2.metric("CGPA", f"{academics.round_2dp(cgpa_val)}",
              help="Set manually in Settings." if cgpa_manual else None)
    k3.metric("Average %", f"{np.mean(percents):.1f}" if percents else "0.0")
    k4.metric("Avg Attendance", f"{np.mean(att_vals):.0f}%" if att_vals else "0%")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Quick start")
            st.markdown(
                "- **📚 Marks** — enter exam scores and see grades update live.\n"
                "- **🕒 Attendance** — track classes and get skip/attend advice.\n"
                "- **📈 History** — save snapshots to build your CGPA over time.\n"
                "- **⚙️ Settings** — customise subjects, credits and exam weightage."
            )
    with col2:
        with st.container(border=True):
            st.subheader("This semester at a glance")
            below = [r for r in summary["rows"]
                     if r["Percent"] < state["semester"]["perf_threshold"]]
            st.write(f"**Subjects:** {len(summary['rows'])}  •  "
                     f"**Credits:** {summary['total_credits']:.0f}")
            if below:
                st.warning(f"{len(below)} subject(s) below "
                           f"{state['semester']['perf_threshold']:.0f}% — check the Marks tab.")
            else:
                st.success("All subjects are above your alert threshold. 🎯")
