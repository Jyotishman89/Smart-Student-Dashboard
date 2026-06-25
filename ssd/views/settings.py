"""Settings: semesters, subjects, exam weightage, threshold, account."""
from __future__ import annotations

import streamlit as st

from .. import academics, auth
from .. import repository as repo
from ..db import session_scope
from . import _common as c


def render() -> None:
    uid = c.require_user()
    sid = c.current_semester_id(uid)
    state = c.load_state(sid)  # Settings always edits live data, never a snapshot

    st.subheader("⚙️ Settings")
    if c.viewing_snapshot_id() is not None:
        cols = st.columns([4, 1])
        cols[0].info("👁️ You're viewing a snapshot elsewhere. Settings still edit your "
                     "live data.")
        if cols[1].button("↩️ Return to live", use_container_width=True):
            c.set_viewing_snapshot(None)
            st.rerun()

    _semester_section(uid, sid, state)
    st.divider()
    _subjects_section(uid, sid, state)
    st.divider()
    _components_section(uid, sid, state)
    st.divider()
    _threshold_section(uid, sid, state)
    st.divider()
    _grades_section(uid, sid, state)
    st.divider()
    _account_section(uid)


# ---------------------------------------------------------------- semesters ---
def _semester_section(uid: int, sid: int, state: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 🎓 Semesters")
        sems = c.semesters(uid)

        col1, col2 = st.columns([2, 1])
        with col1:
            new_label = st.text_input("Rename active semester",
                                      value=state["semester"]["label"], key="sem_label")
            if st.button("Save name"):
                with c.writing() as session:
                    repo.update_semester_meta(session, sid, label=new_label)
                st.toast("Semester renamed.", icon="✏️")
                st.rerun()
        with col2:
            add_label = st.text_input("New semester label", placeholder="Sem-4", key="sem_new")
            if st.button("➕ Add semester"):
                with c.writing() as session:
                    repo.seed_default_semester(session, uid,
                                               label=add_label.strip() or f"Sem-{len(sems)+1}",
                                               make_active=True)
                st.session_state.pop("active_semester_id", None)
                st.toast("Semester added and activated.", icon="🎓")
                st.rerun()

        if len(sems) > 1:
            if st.button("🗑️ Delete active semester", help="Removes this semester and its data."):
                with c.writing() as session:
                    repo.delete_semester(session, uid, sid)
                st.session_state.pop("active_semester_id", None)
                st.toast("Semester deleted.", icon="🗑️")
                st.rerun()


# ---------------------------------------------------------------- subjects ----
def _subjects_section(uid: int, sid: int, state: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 📘 Subjects & Credits")
        st.caption("Add, rename or remove subjects and set their credits.")
        df = c.subjects_dataframe(state)
        edited = st.data_editor(
            df, hide_index=True, use_container_width=True, num_rows="dynamic",
            column_config={
                "id": None,  # hide internal id
                "Subject": st.column_config.TextColumn("Subject", required=True),
                "Credits": st.column_config.NumberColumn("Credits", min_value=0.0, step=0.5),
            },
            key="subjects_editor",
        )
        if st.button("Save subjects", type="primary"):
            rows = [{"id": (None if _is_blank(r.get("id")) else int(r["id"])),
                     "name": r["Subject"], "credits": r.get("Credits", 0)}
                    for r in edited.to_dict("records") if str(r.get("Subject", "")).strip()]
            with c.writing() as session:
                repo.set_subjects(session, sid, rows)
            st.toast("Subjects updated.", icon="📘")
            st.rerun()


# -------------------------------------------------------------- components ----
def _components_section(uid: int, sid: int, state: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 🧮 Exam Weightage (max marks)")
        st.caption("Define how each exam component is weighted. Totals normally sum to 100.")
        df = c.components_dataframe(state)
        edited = st.data_editor(
            df, hide_index=True, use_container_width=True, num_rows="dynamic",
            column_config={
                "id": None,
                "Component": st.column_config.TextColumn("Component", required=True),
                "Max Marks": st.column_config.NumberColumn("Max Marks", min_value=0.0,
                                                           max_value=200.0, step=1.0),
            },
            key="components_editor",
        )
        total = float(edited["Max Marks"].fillna(0).sum())
        if abs(total - 100.0) > 1e-6:
            st.warning(f"Components currently sum to {total:g}. Keeping it at 100 is recommended.")
        else:
            st.success("Components sum to 100. ✅")
        if st.button("Save weightage", type="primary"):
            rows = [{"id": (None if _is_blank(r.get("id")) else int(r["id"])),
                     "name": r["Component"], "max_marks": r.get("Max Marks", 0)}
                    for r in edited.to_dict("records") if str(r.get("Component", "")).strip()]
            with c.writing() as session:
                repo.set_components(session, sid, rows)
            st.toast("Weightage updated and marks re-clamped.", icon="🧮")
            st.rerun()


# --------------------------------------------------------------- threshold ----
def _threshold_section(uid: int, sid: int, state: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 🚩 Performance Alert Threshold")
        value = st.slider(
            "Flag subjects below this percentage on the Marks tab",
            min_value=0, max_value=100,
            value=int(state["semester"]["perf_threshold"]), step=1,
        )
        if value != int(state["semester"]["perf_threshold"]):
            with c.writing() as session:
                repo.update_semester_meta(session, sid, perf_threshold=float(value))
            st.toast(f"Threshold set to {value}%.", icon="🚩")


# ------------------------------------------------------------ sgpa / cgpa -----
def _override_block(*, label: str, slug: str, current, auto: float) -> tuple[str, float] | None:
    """Render one SGPA/CGPA override row.

    Returns ``("save", value)``, ``("reset", 0.0)``, or ``None``. The typed value
    is *sticky*: once saved it stays the shown value even after the "Edit" box is
    unticked. Auto-calculation applies only until a manual value is first saved,
    and "Reset to auto-calculated" is the only way back to auto.
    """
    shown = float(current) if current is not None else auto
    tag = "manual" if current is not None else "auto-calculated"
    st.markdown(f"**{label}:**  {shown:.2f}  _({tag})_")
    st.caption(f"Calculated from marks: {auto:.2f}")
    if st.checkbox(f"Edit {slug.upper()} manually", value=False, key=f"edit_{slug}"):
        val = st.number_input(f"Your {slug.upper()}", min_value=0.0, max_value=10.0,
                              step=0.01, value=shown)
        c1, c2 = st.columns(2)
        if c1.button(f"Save {slug.upper()}", type="primary", key=f"save_{slug}"):
            return ("save", round(float(val), 2))
        if current is not None and c2.button("Reset to auto-calculated",
                                             key=f"reset_{slug}"):
            return ("reset", 0.0)
    elif current is not None:
        if st.button("Reset to auto-calculated", key=f"reset_{slug}_c"):
            return ("reset", 0.0)
    return None


def _grades_section(uid: int, sid: int, state: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 🎯 SGPA / CGPA")
        st.caption("Calculated from your marks by default. Type your own value to "
                   "override (e.g. your official university result) — it stays shown "
                   "even if you untick the box, until you reset it to auto.")

        auto_sgpa = float(academics.round_2dp(repo.summarize_state(state)["sgpa"]))
        cur_sgpa = state["semester"].get("sgpa_override")
        with session_scope() as session:
            user = repo.get_user(session, uid)
            cur_cgpa = user.cgpa_override if user else None
            auto_cgpa_val, _credits = repo.cgpa_auto_for_user(session, uid)
        auto_cgpa = float(academics.round_2dp(auto_cgpa_val))

        action = _override_block(label=f"SGPA — {state['semester']['label']}",
                                 slug="sgpa", current=cur_sgpa, auto=auto_sgpa)
        if action:
            kind, val = action
            with c.writing() as session:
                repo.set_sgpa_override(session, sid, None if kind == "reset" else val)
            st.toast("SGPA setting saved.", icon="🎯")
            st.rerun()

        st.divider()
        action = _override_block(label="CGPA — overall", slug="cgpa",
                                 current=cur_cgpa, auto=auto_cgpa)
        if action:
            kind, val = action
            with c.writing() as session:
                repo.set_cgpa_override(session, uid, None if kind == "reset" else val)
            st.toast("CGPA setting saved.", icon="🎯")
            st.rerun()


# ----------------------------------------------------------------- account ----
def _account_section(uid: int) -> None:
    with st.container(border=True):
        st.markdown("##### 👤 Account")
        st.write(f"**Name:** {st.session_state.get('user_name','—')}")
        st.write(f"**Email:** {st.session_state.get('user_email','—')}")
        if st.button("🔒 Log out"):
            auth.logout()
            st.rerun()


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and value != value) or value == ""
