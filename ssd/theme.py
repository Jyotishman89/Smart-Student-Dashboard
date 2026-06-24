"""Page configuration and global styling (dark "aurora glass" theme).

Styling is intentionally static — no entrance/keyframe animations — because
Streamlit replays CSS on every rerun, which makes entrance animations flicker.
Depth comes from gradients, soft borders and shadows instead.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root{
  --ssd-teal:#00ffd5;
  --ssd-blue:#00a9ff;
  --ssd-card:rgba(255,255,255,0.045);
  --ssd-border:rgba(255,255,255,0.09);
  --ssd-muted:#9fb0c9;
}

/* ---- app background --------------------------------------------------- */
.stApp{
  background:
    radial-gradient(1100px 560px at 8% 4%, rgba(0,255,224,0.07), transparent 42%),
    radial-gradient(1000px 520px at 92% 12%, rgba(0,112,255,0.07), transparent 42%),
    linear-gradient(160deg,#0a111e 0%,#0f172a 55%,#0a111e 100%);
}
.block-container{ padding-top:1.4rem !important; max-width:1200px; }
h1,h2,h3{ letter-spacing:-0.01em; }

/* ---- glass containers ------------------------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--ssd-card);
  border:1px solid var(--ssd-border);
  border-radius:16px;
  box-shadow:0 10px 30px rgba(0,0,0,0.28);
}

/* ---- metric cards ----------------------------------------------------- */
div[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border:1px solid var(--ssd-border);
  border-radius:14px;
  padding:14px 16px;
  box-shadow:0 8px 22px rgba(0,0,0,0.25);
  transition:border-color .15s ease, transform .15s ease;
}
div[data-testid="stMetric"]:hover{
  border-color:rgba(0,255,213,0.28);
  transform:translateY(-2px);
}
div[data-testid="stMetricLabel"]{ color:var(--ssd-muted); font-weight:600; }
div[data-testid="stMetricValue"]{
  font-weight:800;
  background:linear-gradient(90deg,var(--ssd-teal),var(--ssd-blue));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}

/* ---- buttons ---------------------------------------------------------- */
.stButton>button, .stDownloadButton>button{
  border-radius:12px; font-weight:600; color:#e6edf6;
  background:rgba(255,255,255,0.05); border:1px solid var(--ssd-border);
  transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-1px); border-color:rgba(0,255,213,0.35);
}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:linear-gradient(90deg,var(--ssd-teal),var(--ssd-blue));
  color:#001321; border:0; font-weight:700;
  box-shadow:0 8px 20px rgba(0,255,213,0.22);
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  transform:translateY(-1px); box-shadow:0 12px 28px rgba(0,255,213,0.34);
}

/* ---- inputs ----------------------------------------------------------- */
[data-baseweb="input"], [data-baseweb="select"]>div, .stTextArea textarea{
  border-radius:12px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]>div:focus-within{
  border-color:rgba(0,255,213,0.5) !important;
  box-shadow:0 0 0 2px rgba(0,255,213,0.15) !important;
}

/* ---- tabs ------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{ border-radius:10px 10px 0 0; padding:8px 14px; }
.stTabs [aria-selected="true"]{ color:var(--ssd-teal); }

/* ---- sidebar ---------------------------------------------------------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  border-right:1px solid var(--ssd-border);
}

/* ---- hero ------------------------------------------------------------- */
.hero{
  padding:26px 28px; border-radius:20px;
  background:
    radial-gradient(820px 420px at 18% 8%, rgba(0,255,224,0.12), transparent 42%),
    radial-gradient(820px 420px at 82% 18%, rgba(0,112,255,0.12), transparent 42%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border:1px solid var(--ssd-border);
  box-shadow:0 18px 50px rgba(0,0,0,0.35);
}
.hero h1{
  margin:.55rem 0 .2rem 0; font-size:2.3rem; font-weight:800; line-height:1.12;
  background:linear-gradient(90deg,#eafff9,#bfe9ff);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{ color:var(--ssd-muted); margin:.1rem 0 .8rem 0; font-size:1.02rem; }
.pill{
  display:inline-flex; gap:.5rem; align-items:center; padding:.4rem .9rem;
  border-radius:999px; font-weight:600; font-size:.85rem; color:#cffaf0;
  background:rgba(0,255,224,0.12); border:1px solid rgba(0,255,224,0.28);
}
.chip{
  display:inline-flex; align-items:center; padding:.3rem .7rem; border-radius:999px;
  background:rgba(255,255,255,0.06); border:1px solid var(--ssd-border);
  font-size:.8rem; color:#dbe6f5; margin:.15rem .35rem .15rem 0;
  transition:border-color .15s ease;
}
.chip:hover{ border-color:rgba(0,255,213,0.30); }

/* ---- misc polish ------------------------------------------------------ */
footer{ visibility:hidden; }
hr{ border-color:var(--ssd-border); }
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
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div>{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
