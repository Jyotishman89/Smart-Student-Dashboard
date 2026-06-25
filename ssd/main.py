"""Application entrypoint: page config, auth gate, navigation, sidebar."""
from __future__ import annotations

import streamlit as st

from . import auth
from . import repository as repo
from .db import init_db, session_scope
from .theme import configure_page
from .views import _common as common
from .views import attendance, auth_view, history, home, marks, settings


@st.cache_resource(show_spinner=False)
def _db_ready() -> bool:
    """Ensure tables exist — once per server process.

    ``init_db()`` issues table-reflection queries; calling it on every rerun
    cost a round-trip to the remote (Neon) database on each interaction, which
    showed up as lag. Caching makes it a one-time startup cost.
    """
    init_db()
    return True


def _sidebar(user_id: int) -> None:
    with st.sidebar:
        st.markdown(f"### 👋 {st.session_state.get('user_name', 'Student')}")
        st.caption(st.session_state.get("user_email", ""))

        with session_scope() as session:
            sems = [(s.id, s.label) for s in repo.list_semesters(session, user_id)]
            active = repo.get_active_semester(session, user_id)
            active_id = st.session_state.get("active_semester_id", active.id)

        ids = [sid for sid, _ in sems]
        labels = {sid: label for sid, label in sems}
        if active_id not in ids:
            active_id = ids[0]
        chosen = st.selectbox(
            "Active semester", ids, index=ids.index(active_id),
            format_func=lambda sid: labels.get(sid, str(sid)),
        )
        if chosen != active_id:
            st.session_state.active_semester_id = chosen
            common.set_viewing_snapshot(None)  # snapshots are per-semester
            with session_scope() as session:
                repo.set_active_semester(session, user_id, chosen)
            st.rerun()
        st.session_state.active_semester_id = chosen
        st.divider()


def run() -> None:
    configure_page()
    _db_ready()  # tables created once per process, not on every rerun

    user_id = auth.restore_session()
    if not user_id:
        auth_view.render()
        return

    _sidebar(user_id)

    pages = [
        st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
        st.Page(marks.render, title="Marks", icon="📚", url_path="marks"),
        st.Page(attendance.render, title="Attendance", icon="🕒", url_path="attendance"),
        st.Page(history.render, title="History / CGPA", icon="📈", url_path="history"),
        st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
    ]
    st.navigation(pages).run()
