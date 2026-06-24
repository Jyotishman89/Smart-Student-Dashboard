"""History, snapshots, SGPA/CGPA trends."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import academics, charts
from .. import repository as repo
from ..db import session_scope
from . import _common as c


def _snapshots_as_dicts(user_id: int) -> list[dict]:
    with session_scope() as session:
        snaps = repo.list_snapshots(session, user_id)
        return [{
            "id": s.id, "taken_at": s.taken_at, "label": s.semester_label,
            "sgpa": s.sgpa, "total_credits": s.total_credits, "payload": s.payload or {},
        } for s in snaps]


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state = c.load_state(sid)
    summary = repo.summarize_state(state)

    st.subheader("📈 History & CGPA")

    with session_scope() as session:
        cgpa_val, cgpa_credits = repo.cgpa_for_user(session, uid)

    k = st.columns(3)
    k[0].metric("Current SGPA", f"{academics.round_2dp(summary['sgpa'])}")
    k[1].metric("Semester Credits", f"{summary['total_credits']:.0f}")
    k[2].metric("CGPA (latest per semester)", f"{academics.round_2dp(cgpa_val)}")

    # ----- actions -----
    a1, a2 = st.columns([1, 2])
    with a1:
        if st.button("💾 Save snapshot", type="primary", use_container_width=True):
            with session_scope() as session:
                repo.create_snapshot(session, uid, sid)
            st.toast("Snapshot saved.", icon="💾")
            st.rerun()

    snaps = _snapshots_as_dicts(uid)
    if not snaps:
        st.info("No snapshots yet. Save one to start building your CGPA timeline.")
        return

    with a2:
        labels = {f"{s['taken_at']:%Y-%m-%d %H:%M} — {s['label']} "
                  f"(SGPA {academics.round_2dp_from_float(s['sgpa'])})": s["id"]
                  for s in snaps}
        pick = st.selectbox("Restore a snapshot into the active semester", list(labels))
        if st.button("↩️ Restore selected", use_container_width=True):
            with session_scope() as session:
                repo.restore_snapshot(session, sid, labels[pick])
            st.toast("Snapshot restored into the active semester.", icon="↩️")
            st.rerun()

    # ----- download -----
    hist_df = pd.DataFrame([
        {"Timestamp": s["taken_at"], "Semester": s["label"],
         "SGPA": s["sgpa"], "TotalCredits": s["total_credits"]}
        for s in snaps
    ])
    buf = io.StringIO()
    hist_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download history (CSV)", data=buf.getvalue(),
                       file_name="history.csv", mime="text/csv")

    # ----- SGPA over time -----
    tdf = hist_df.sort_values("Timestamp")
    st.plotly_chart(
        charts.time_series(tdf["Timestamp"], tdf["SGPA"], "SGPA Over Time", y_max=10),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ----- per-subject trend -----
    subj_names = sorted({sub["name"] for s in snaps for sub in s["payload"].get("subjects", [])})
    if subj_names:
        chosen = st.selectbox("Per-subject trend", subj_names)
        points = []
        for s in snaps:
            for sub in s["payload"].get("subjects", []):
                if sub["name"] == chosen:
                    total = academics.subject_total(sub.get("scores", {}).values())
                    points.append({"Timestamp": s["taken_at"], "Total": total})
        if points:
            pdf = pd.DataFrame(points).sort_values("Timestamp")
            st.plotly_chart(
                charts.time_series(pdf["Timestamp"], pdf["Total"],
                                   f"{chosen} — Total Marks Over Time"),
                use_container_width=True, config={"displayModeBar": False},
            )
