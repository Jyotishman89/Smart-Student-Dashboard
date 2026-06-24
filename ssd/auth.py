"""Authentication: signup, login, and cookie-persisted sessions.

Passwords are hashed with bcrypt. The login session is kept in
``st.session_state`` and mirrored into a signed cookie (HMAC over the user id,
using COOKIE_SECRET) so a browser refresh does not log the user out.
"""
from __future__ import annotations

import hashlib
import hmac
import re

import bcrypt

from . import config
from . import repository as repo
from .db import session_scope

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_COOKIE_NAME = "ssd_session"


# ----------------------------------------------------------------- hashing ----
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- tokens ------
def _sign(user_id: int) -> str:
    msg = str(user_id).encode("utf-8")
    sig = hmac.new(config.cookie_secret().encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return f"{user_id}.{sig}"


def _verify_token(token: str) -> int | None:
    try:
        uid_str, sig = token.split(".", 1)
        expected = _sign(int(uid_str)).split(".", 1)[1]
        if hmac.compare_digest(sig, expected):
            return int(uid_str)
    except (ValueError, AttributeError):
        pass
    return None


# ------------------------------------------------------------ cookie helper ---
def _cookie_manager():
    """One CookieManager per script run, stashed in session_state."""
    import streamlit as st
    if "_cookie_mgr" not in st.session_state:
        import extra_streamlit_components as stx
        st.session_state._cookie_mgr = stx.CookieManager(key="ssd_cookie_mgr")
    return st.session_state._cookie_mgr


def _set_cookie(user_id: int) -> None:
    try:
        from datetime import datetime, timedelta
        _cookie_manager().set(
            _COOKIE_NAME, _sign(user_id),
            expires_at=datetime.now() + timedelta(days=14), key="set_session_cookie",
        )
    except Exception:
        pass  # cookie persistence is best-effort; session_state still works


def _clear_cookie() -> None:
    try:
        _cookie_manager().delete(_COOKIE_NAME, key="del_session_cookie")
    except Exception:
        pass


# ------------------------------------------------------------ public API ------
def signup(email: str, full_name: str, roll_no: str,
           password: str, confirm: str) -> tuple[bool, str]:
    email = email.lower().strip()
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < config.MIN_PASSWORD_LEN:
        return False, f"Password must be at least {config.MIN_PASSWORD_LEN} characters."
    if password != confirm:
        return False, "Passwords do not match."
    if not full_name.strip():
        return False, "Please enter your name."
    with session_scope() as session:
        if repo.get_user_by_email(session, email):
            return False, "An account with that email already exists."
        repo.create_user(
            session, email=email, password_hash=hash_password(password),
            full_name=full_name, roll_no=roll_no,
        )
    return True, "Account created. You can log in now."


def login(email: str, password: str) -> tuple[bool, str]:
    import streamlit as st
    email = email.lower().strip()
    with session_scope() as session:
        user = repo.get_user_by_email(session, email)
        if user is None or not verify_password(password, user.password_hash):
            return False, "Invalid email or password."
        st.session_state.user_id = user.id
        st.session_state.user_name = user.full_name
        st.session_state.user_email = user.email
    _set_cookie(st.session_state.user_id)
    return True, "Logged in."


def logout() -> None:
    import streamlit as st
    _clear_cookie()
    for key in ("user_id", "user_name", "user_email", "active_semester_id"):
        st.session_state.pop(key, None)


def restore_session() -> int | None:
    """Return the logged-in user id, restoring from the cookie if needed."""
    import streamlit as st
    if st.session_state.get("user_id"):
        return st.session_state.user_id
    try:
        token = _cookie_manager().get(_COOKIE_NAME)
    except Exception:
        token = None
    if token:
        uid = _verify_token(token)
        if uid:
            with session_scope() as session:
                user = repo.get_user(session, uid)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.user_name = user.full_name
                    st.session_state.user_email = user.email
                    return user.id
    return None


def current_user_id() -> int | None:
    import streamlit as st
    return st.session_state.get("user_id")
