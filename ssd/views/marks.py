"""Marks & exams page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts
from .. import repository as repo
from ..config import PASS_PERCENT, TARGET_PERCENT
from ..db import session_scope
from . import _common as c


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state = c.load_state(sid)

    st.subheader("📚 Marks & Exams")
    if not state["subjects"]:
        st.info("No subjects yet.")
        return

    top = st.columns([1, 2])
    with top[0]:
        chart_kind = c.chart_type("marks_chart_type")
    with top[1]:
        st.caption(f"Pass line at {PASS_PERCENT:g}% • Target line at {TARGET_PERCENT:g}%. "
                   "Edit marks or rename a subject in the grid below — changes save "
                   "automatically.")

    # ----- editable grid (rename subjects inline + edit marks) -----
    df = c.marks_dataframe(state)
    col_cfg = {"Subject": st.column_config.TextColumn(
        "Subject", required=True, help="Rename a subject by editing it here."
    )}
    for comp in state["components"]:
        col_cfg[comp["name"]] = st.column_config.NumberColumn(
            f"{comp['name']} ({comp['max_marks']:g})",
            min_value=0.0, max_value=float(comp["max_marks"]), step=0.5,
        )
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, num_rows="fixed",
        column_config=col_cfg, key="marks_editor",
    )

    # ----- persist only when something actually changed (fixes constant-write bug) -----
    comp_names = [comp["name"] for comp in state["components"]]
    name_updates: dict[int, str] = {}
    score_updates: dict[int, dict[str, float]] = {}
    for subj, (_, row) in zip(state["subjects"], edited.iterrows(), strict=False):
        new_name = "" if pd.isna(row["Subject"]) else str(row["Subject"]).strip()
        if new_name and new_name != subj["name"]:
            name_updates[subj["id"]] = new_name
        score_updates[subj["id"]] = {cn: float(row[cn]) for cn in comp_names}

    scores_changed = not edited[comp_names].equals(df[comp_names])
    if name_updates or scores_changed:
        with session_scope() as session:
            if name_updates:
                repo.rename_subjects(session, sid, name_updates)
            if scores_changed:
                repo.save_scores(session, sid, score_updates)
        state = c.load_state(sid)  # reload renamed/clamped values

    summary = repo.summarize_state(state)
    rows = pd.DataFrame(summary["rows"])

    # ----- KPIs -----
    pct = rows["Percent"].to_numpy()
    m = st.columns(5)
    m[0].metric("Average %", f"{np.mean(pct):.1f}" if len(pct) else "0.0")
    m[1].metric("Median %", f"{np.median(pct):.1f}" if len(pct) else "0.0")
    m[2].metric("Best %", f"{np.max(pct):.1f}" if len(pct) else "0.0")
    m[3].metric("Worst %", f"{np.min(pct):.1f}" if len(pct) else "0.0")
    m[4].metric("Std Dev", f"{np.std(pct, ddof=1):.1f}" if len(pct) > 1 else "0.0")

    # ----- performance alerts -----
    threshold = state["semester"]["perf_threshold"]
    below = rows[rows["Percent"] < threshold]
    if not below.empty:
        st.error(f"⚠️ Below {threshold:.0f}%:")
        st.dataframe(below[["Subject", "Total", "Percent", "Grade"]],
                     hide_index=True, use_container_width=True)
    else:
        st.success(f"✅ All subjects are at or above {threshold:.0f}%.")

    # ----- chart -----
    labels = rows["Subject"].tolist()
    if chart_kind == "Bar":
        fig = charts.bar(labels, rows["Percent"], "Marks by Subject (%)",
                         y_max=100, pass_line=PASS_PERCENT, target_line=TARGET_PERCENT)
    elif chart_kind == "Pie":
        fig = charts.pie(labels, rows["Total"], "Share of Total Marks")
    else:
        fig = charts.line(labels, rows["Percent"], "Marks Trend (%)", y_max=100)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ----- detailed results -----
    with st.expander("Detailed results"):
        st.dataframe(
            rows[["Subject", "Total", "Max", "Percent", "Grade", "Result", "Credits"]],
            hide_index=True, use_container_width=True,
        )
