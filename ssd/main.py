"""Application entrypoint: page config, auth gate, navigation, sidebar."""
from __future__ import annotations

import streamlit as st

from . import auth
from . import repository as repo
from .db import init_db
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

        sems = common.semesters(user_id)  # cached: [(id, label, is_active)]
        ids = [s[0] for s in sems]
        labels = {s[0]: s[1] for s in sems}
        active_id = st.session_state.get("active_semester_id")
        if active_id not in ids:
            active_id = next((s[0] for s in sems if s[2]), ids[0])

        chosen = st.selectbox(
            "Active semester", ids, index=ids.index(active_id),
            format_func=lambda sid: labels.get(sid, str(sid)),
        )
        if chosen != active_id:
            st.session_state.active_semester_id = chosen
            common.set_viewing_snapshot(None)  # snapshots are per-semester
            with common.writing() as session:
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
