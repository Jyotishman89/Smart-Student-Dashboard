"""Login / signup screen shown when no user is authenticated."""
from __future__ import annotations

import streamlit as st

from .. import auth
from ..theme import hero


def render() -> None:
    hero(
        "Smart Student Dashboard",
        "Track marks, attendance, SGPA & CGPA — securely, across semesters.",
        chips=["Multi-user", "Secure login", "Cloud database", "Dynamic charts"],
    )
    st.write("")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Why you'll like it")
            st.markdown(
                "- 🔐 **Private accounts** — your data is yours, bcrypt-secured.\n"
                "- 📚 **Live grades** — edit marks and watch grades & SGPA update.\n"
                "- 🕒 **Attendance advice** — know if you can skip the next class.\n"
                "- 📈 **CGPA timeline** — snapshot each semester and track trends."
            )
            st.caption("🔒 Passwords are hashed with bcrypt and never stored in plain text.")

    with right:
        with st.container(border=True):
            login_tab, signup_tab = st.tabs(["🔐 Log in", "✨ Create account"])

            with login_tab:
                with st.form("login_form"):
                    roll_no = st.text_input("Roll number", placeholder="e.g. CSB24009")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Log in", type="primary",
                                                      use_container_width=True)
                if submitted:
                    ok, msg = auth.login(roll_no, password)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            with signup_tab:
                with st.form("signup_form"):
                    full_name = st.text_input("Full name")
                    roll_no = st.text_input("Roll number", placeholder="e.g. 23CSE102")
                    email = st.text_input("Email ", placeholder="you@example.com")
                    pw = st.text_input("Password ", type="password",
                                       help="At least 8 characters.")
                    confirm = st.text_input("Confirm password", type="password")
                    submitted = st.form_submit_button("Create account", type="primary",
                                                      use_container_width=True)
                if submitted:
                    ok, msg = auth.signup(email, full_name, roll_no, pw, confirm)
                    if ok:
                        st.success(msg + " Switch to the **Log in** tab.")
                    else:
                        st.error(msg)
