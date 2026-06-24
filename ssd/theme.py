"""Page configuration and global styling (dark + subtle glassmorphism)."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
.stApp {
  background:
    radial-gradient(1200px 600px at 10% 10%, rgba(0,255,224,0.06), transparent 40%),
    radial-gradient(1000px 500px at 90% 20%, rgba(0,112,255,0.06), transparent 40%),
    linear-gradient(135deg, #0b1220 0%, #0f172a 55%, #0b1220 100%);
}
.block-container { padding-top: 1.2rem !important; }

/* Native bordered containers get the glass look */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,0.04);
  border-radius: 16px;
}

/* Primary buttons */
.stButton>button[kind="primary"], .stFormSubmitButton>button {
  background: linear-gradient(90deg,#00ffd5,#00a9ff);
  color:#001321; border:0; font-weight:700;
  box-shadow: 0 8px 18px rgba(0,255,213,0.20);
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover {
  box-shadow: 0 12px 26px rgba(0,255,213,0.32);
}

.hero {
  padding: 22px; border-radius: 18px;
  background:
    radial-gradient(800px 400px at 20% 10%, rgba(0,255,224,0.10), transparent 40%),
    radial-gradient(800px 400px at 80% 20%, rgba(0,112,255,0.10), transparent 40%),
    rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.08);
}
.pill {
  display:inline-flex; gap:.5rem; align-items:center; padding:.4rem .85rem;
  border-radius:999px; font-weight:600; font-size:.85rem;
  background:rgba(0,255,224,0.12); border:1px solid rgba(0,255,224,0.25);
}
.chip {
  padding:.22rem .55rem; border-radius:999px; background:rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.12); font-size:.78rem; margin-right:.3rem;
}
</style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title="Smart Student Dashboard",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(
        f"""
        <div class="hero">
          <span class="pill">🎓 Student Analytics</span>
          <h1 style="margin:.5rem 0 .15rem 0;">{title}</h1>
          <p style="opacity:.85; margin:.1rem 0 .6rem 0;">{subtitle}</p>
          <div>{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
