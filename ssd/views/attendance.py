"""Attendance tracker page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import academics, charts
from .. import repository as repo
from ..config import ATTENDANCE_REQ
from ..db import session_scope
from . import _common as c


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state = c.load_state(sid)

    st.subheader("🕒 Attendance Tracker")
    if not state["subjects"]:
        st.info("No subjects yet. Add some in **⚙️ Settings**.")
        return

    top = st.columns([1, 2])
    with top[0]:
        chart_kind = c.chart_type("att_chart_type")
    with top[1]:
        st.caption(f"Minimum required attendance: {ATTENDANCE_REQ:g}%. "
                   "Edit the grid below — changes save automatically.")

    df = c.attendance_dataframe(state)
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={
            "Subject": st.column_config.TextColumn("Subject", disabled=True),
            "Classes Held": st.column_config.NumberColumn("Classes Held", min_value=0, step=1),
            "Classes Attended": st.column_config.NumberColumn("Classes Attended",
                                                              min_value=0, step=1),
        },
        key="att_editor",
    )

    if not edited[["Classes Held", "Classes Attended"]].equals(
        df[["Classes Held", "Classes Attended"]]
    ):
        updates = {}
        for subj, (_, row) in zip(state["subjects"], edited.iterrows(), strict=False):
            updates[subj["id"]] = (int(row["Classes Held"]), int(row["Classes Attended"]))
        with session_scope() as session:
            repo.save_attendance(session, updates)
        state = c.load_state(sid)

    # ----- compute advice -----
    records = []
    for s in state["subjects"]:
        held, attended = s["attendance"]["held"], s["attendance"]["attended"]
        cur, skip_p, attend_p, msg = academics.next_class_advice(held, attended)
        records.append({
            "Subject": s["name"], "Classes Held": held, "Classes Attended": attended,
            "Current %": cur, "If Skip Next %": skip_p, "If Attend Next %": attend_p,
            "Advice": msg,
        })
    out = pd.DataFrame(records)

    avg_att = out["Current %"].mean() if not out.empty else 0.0
    n_below = int((out["Current %"] < ATTENDANCE_REQ).sum())
    k = st.columns(3)
    k[0].metric("Average Attendance", f"{avg_att:.0f}%")
    k[1].metric("Subjects Below 75%", f"{n_below}")
    k[2].metric("Subjects Tracked", f"{len(out)}")

    labels = out["Subject"].tolist()
    if chart_kind == "Bar":
        fig = charts.bar(labels, out["Current %"], "Attendance by Subject (%)",
                         y_max=100, pass_line=ATTENDANCE_REQ)
    elif chart_kind == "Pie":
        fig = charts.pie(labels, out["Classes Attended"], "Attended Classes Share")
    else:
        fig = charts.line(labels, out["Current %"], "Attendance Trend (%)", y_max=100)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("##### Recommendation per subject")
    st.dataframe(out, hide_index=True, use_container_width=True)
