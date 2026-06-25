"""History, snapshots, SGPA/CGPA trends."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import academics, charts
from .. import repository as repo
from ..db import session_scope
from . import _common as c


def _snapshots_as_dicts(user_id: int, semester_id: int) -> list[dict]:
    with session_scope() as session:
        snaps = repo.list_snapshots(session, user_id, semester_id)
        return [{
            "id": s.id, "taken_at": s.taken_at, "label": s.semester_label,
            "sgpa": s.sgpa, "total_credits": s.total_credits, "payload": s.payload or {},
        } for s in snaps]


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state, viewing = c.page_state(uid, sid)
    summary = repo.summarize_state(state)

    st.subheader("📈 History & CGPA")

    snaps = _snapshots_as_dicts(uid, sid)
    with session_scope() as session:
        cgpa_val, cgpa_credits, cgpa_manual = repo.cgpa_for_user(session, uid)

    # ----- switch the WHOLE app between live data and any saved snapshot -----
    # Selecting a snapshot here puts every page into a read-only view of that
    # snapshot; "Current (live)" returns to your real data. It never writes to
    # the database, so marks and manual SGPA/CGPA settings stay untouched.
    LIVE = "Current (live)"
    options: dict[str, int | None] = {LIVE: None}
    for s in snaps:
        options[f"{s['taken_at']:%Y-%m-%d %H:%M}  ·  SGPA "
                f"{academics.round_2dp_from_float(s['sgpa'])}"] = s["id"]
    if snaps:
        labels = list(options)
        current = next((lbl for lbl, snap_id in options.items()
                        if snap_id == c.viewing_snapshot_id()), LIVE)
        pick = st.selectbox(
            f"View — {state['semester']['label']}", labels,
            index=labels.index(current), key=f"view_{sid}",
            help="Switch the whole app between your live data and any saved "
                 "snapshot. Viewing a snapshot is read-only and never changes "
                 "your marks or manual settings.",
        )
        if options[pick] != c.viewing_snapshot_id():
            c.set_viewing_snapshot(options[pick])
            st.rerun()

    c.view_banner(viewing)

    # ----- metric cards reflect the current view (live or snapshot) -----
    k = st.columns(3)
    k[0].metric("SGPA", f"{academics.round_2dp(summary['sgpa_effective'])}",
                help=(f"Snapshot from {viewing['taken_at']:%Y-%m-%d %H:%M}." if viewing
                      else ("Set manually in Settings." if summary["sgpa_is_manual"]
                            else None)))
    k[1].metric("Semester Credits", f"{summary['total_credits']:.0f}")
    k[2].metric("CGPA (latest per semester)", f"{academics.round_2dp(cgpa_val)}",
                help="Set manually in Settings." if cgpa_manual else None)

    # ----- actions -----
    a1, a2 = st.columns(2)
    with a1:
        if st.button("💾 Save snapshot", type="primary", use_container_width=True,
                     disabled=viewing is not None,
                     help=("Return to live to save." if viewing is not None
                           else "Save your current live data as a snapshot.")):
            with session_scope() as session:
                repo.create_snapshot(session, uid, sid)
            st.toast("Snapshot saved.", icon="💾")
            st.rerun()
    with a2:
        if st.button(f"↩️ Restore into {state['semester']['label']}",
                     use_container_width=True, disabled=viewing is None,
                     help=("Overwrites this semester's live marks with the snapshot "
                           "you're viewing." if viewing is not None
                           else "Select a snapshot above to enable restore.")):
            with session_scope() as session:
                repo.restore_snapshot(session, sid, viewing["id"])
            c.set_viewing_snapshot(None)  # the snapshot is now your live data
            st.toast("Snapshot restored into your live data. "
                     "Manual SGPA/CGPA settings were left unchanged.", icon="↩️")
            st.rerun()

    if not snaps:
        st.info(f"No snapshots yet for {state['semester']['label']}. "
                "Save one to start building this semester's history.")
        return

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
