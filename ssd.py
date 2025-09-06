import os
from datetime import datetime
from decimal import Decimal, getcontext, ROUND_HALF_UP

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json

# ===================== CONFIG =====================
st.set_page_config(page_title="Student Performance Portal", page_icon="🎓", layout="wide")

# ===================== CONSTANTS =====================
PASSWORD = "jyotishman24"  # one-time password (asked only on Home)

SUBJECTS = [
    "Data Structures", "DS Lab", "Mathematics III", "Discrete Mathematics",
    "IT Workshop", "Idea Lab", "Digital Logic Design", "DLD Lab"
]
COMPONENTS = [("Sessional 1", 10), ("Mid Term", 30), ("Sessional 2", 10), ("End Term", 50)]
PASS_MARK = 40
TARGET_MARK = 75
ATT_REQ = 75.0

DEFAULT_CREDITS = {
    "Data Structures": 4, "DS Lab": 1, "Mathematics III": 4, "Discrete Mathematics": 3,
    "IT Workshop": 1, "Idea Lab": 2, "Digital Logic Design": 3, "DLD Lab": 1
}
GRADE_POINTS = {"A+":10, "A":9, "B+":8, "B":7, "C":6, "D":5, "F":0}

DATA_DIR = "student_data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
os.makedirs(DATA_DIR, exist_ok=True)


# Precise decimals for SGPA/CGPA
getcontext().prec = 28

# ===================== THEME / CSS (Dark + Glass) =====================
DARK_CSS = """
<style>
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(0,255,224,0.08), transparent 40%),
              radial-gradient(1000px 500px at 90% 20%, rgba(0,112,255,0.08), transparent 40%),
              linear-gradient(135deg, #0b1220 0%, #0f172a 55%, #0b1220 100%);
  color: #e6edf6;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
}
.block-container { padding-top: 0.8rem !important; }
.glass {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  backdrop-filter: blur(12px);
  border-radius: 20px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.35);
  transition: transform .2s ease, box-shadow .2s ease, border .2s ease;
}
.glass:hover {
  transform: translateY(-4px) scale(1.01);
  box-shadow: 0 16px 40px rgba(0,0,0,0.45);
  border-color: rgba(0,255,224,0.25);
}
.stButton>button {
  background: linear-gradient(90deg,#00ffd5,#00a9ff);
  color:#001321; border:0; border-radius: 12px; font-weight:700;
  box-shadow: 0 8px 18px rgba(0,255,213,0.25);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(0,255,213,0.36); }
.stButton>button:active { transform: translateY(0); }
@keyframes floaty { 0%{transform:translateY(0)} 50%{transform:translateY(-6px)} 100%{transform:translateY(0)} }
.hero {
  padding: 22px; border-radius: 20px; animation: floaty 6s ease-in-out infinite;
  background: radial-gradient(800px 400px at 20% 10%, rgba(0,255,224,0.09), transparent 40%),
              radial-gradient(800px 400px at 80% 20%, rgba(0,112,255,0.09), transparent 40%),
              rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.pill {
  display:inline-flex; gap:.5rem; align-items:center; padding:.45rem .9rem; border-radius:999px; font-weight:600;
  background:rgba(0,255,224,0.12); border:1px solid rgba(0,255,224,0.25);
}
.dataframe tbody tr:hover { background: rgba(0,255,224,0.06) !important; }
.chip { padding:.25rem .55rem; border-radius:999px; background:rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.12); font-size:.8rem; }
.selector-row { display:flex; align-items:center; gap:.5rem; margin:.25rem 0 .75rem 0; }

/* Roll no on top-right */
.top-right {
  position: sticky; top: 0; z-index: 10;
  text-align: right; padding: 6px 2px 8px 2px; margin-top: -8px;
  color: #a8b3cf; font-size: 14px;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ===================== SESSION STATE =====================
def _default_marks_df():
    return pd.DataFrame({
        "Subject": SUBJECTS,
        "Sessional 1": [7]*8, "Mid Term": [21]*8, "Sessional 2": [7]*8, "End Term": [35]*8
    })

def _default_att_df():
    return pd.DataFrame({
        "Subject": SUBJECTS,
        "Classes Held": [28]*8, "Classes Attended": [22]*8
    })

if "authed" not in st.session_state: st.session_state.authed = False
if "roll_no" not in st.session_state: st.session_state.roll_no = ""  # NEW

if "semester" not in st.session_state: st.session_state.semester = "Sem-3"
if "marks" not in st.session_state: st.session_state.marks = _default_marks_df()
if "attendance" not in st.session_state: st.session_state.attendance = _default_att_df()
if "credits" not in st.session_state:
    st.session_state.credits = pd.DataFrame({"Subject": SUBJECTS, "Credits": [DEFAULT_CREDITS[s] for s in SUBJECTS]})

if "last_totals" not in st.session_state: st.session_state.last_totals = np.zeros(len(SUBJECTS))
if "last_att" not in st.session_state: st.session_state.last_att = np.zeros(len(SUBJECTS))

# ==== NEW: Performance threshold (for alerts) ====
if "perf_threshold" not in st.session_state:
    st.session_state.perf_threshold = 50.0  # you can change the default

# ==== NEW: Editable exam weights (max marks per component) ====
if "component_max" not in st.session_state:
    # Initialize from your original COMPONENTS list
    st.session_state.component_max = {name: mx for name, mx in COMPONENTS}



# ===================== HELPERS =====================
def clamp_marks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cm = st.session_state.component_max  # live weights
    for comp, mx in cm.items():
        if comp in out.columns:
            out[comp] = pd.to_numeric(out[comp], errors="coerce").fillna(0).clip(0, float(mx))
    return out


def subject_totals(df_marks: pd.DataFrame) -> np.ndarray:
    cols = [c for c, _ in COMPONENTS]
    return df_marks[cols].astype(float).sum(axis=1).to_numpy()

def grade_from_total(x: float) -> str:
    if x >= 90: return "A+"
    if x >= 80: return "A"
    if x >= 70: return "B+"
    if x >= 60: return "B"
    if x >= 50: return "C"
    if x >= 40: return "D"
    return "F"

def att_percent(held: int, attended: int) -> float:
    held = max(0, int(held))
    attended = max(0, min(int(attended), held))
    return 100.0 * attended / held if held > 0 else (100.0 if attended > 0 else 0.0)

def next_class_advice(held: int, attended: int, req=ATT_REQ):
    cur = att_percent(held, attended)
    skip_p = att_percent(held + 1, attended)
    attend_p = att_percent(held + 1, attended + 1)
    if skip_p >= req:
        msg = "✅ You can skip the next class and remain above the limit."
    elif attend_p >= req:
        msg = "⚠️ Attend the next class to stay/return above the limit."
    else:
        msg = "❌ Attend the next class. Skipping will worsen your percentage."
    return round(cur, 2), round(skip_p, 2), round(attend_p, 2), msg

# --- Decimal SGPA (exact rounding to show 8.02 when appropriate) ---
def sgpa_calc(df_totals: pd.DataFrame, df_credits: pd.DataFrame):
    m = df_totals[["Subject", "Total", "Grade"]].merge(df_credits, on="Subject", how="left")
    m["Credits"] = pd.to_numeric(m["Credits"], errors="coerce").fillna(0)
    m["GP"] = m["Grade"].map(GRADE_POINTS).fillna(0).astype(int)
    w_sum = Decimal(0)
    c_sum = Decimal(0)
    for _, r in m.iterrows():
        gp = Decimal(int(r["GP"]))
        cr = Decimal(str(r["Credits"]))
        w_sum += gp * cr
        c_sum += cr
    sgpa = Decimal(0) if c_sum == 0 else (w_sum / c_sum).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return sgpa, float(c_sum)

def load_history():
    return pd.read_csv(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame()

def latest_per_semester(df_hist: pd.DataFrame):
    if df_hist.empty: return df_hist
    df = df_hist.copy()
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
    idx = df.sort_values("Timestamp_dt").groupby("Semester")["Timestamp_dt"].idxmax()
    return df.loc[idx].sort_values("Semester")

def cgpa_from_history(df_hist: pd.DataFrame):
    if df_hist.empty: return Decimal(0), Decimal(0)
    last = latest_per_semester(df_hist)
    if last.empty: return Decimal(0), Decimal(0)
    w = [Decimal(str(x)) for x in last["TotalCredits"].astype(float).values]
    s = [Decimal(str(x)) for x in last["SGPA"].astype(float).values]
    tot_cred = sum(w)
    cgpa = Decimal(0) if tot_cred == 0 else (sum([wi*si for wi,si in zip(w,s)]) / tot_cred).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return cgpa, tot_cred

def save_snapshot(semester, df_marks, df_att, df_cred, df_totals, sgpa, tot_cred):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": ts,
        "Semester": semester,
        "RollNo": st.session_state.roll_no,  # NEW
        "SGPA": float(sgpa),
        "TotalCredits": float(tot_cred)
    }
    for _, r in df_totals.iterrows():
        subj = r["Subject"]
        row[f"{subj}__Total"] = float(r["Total"])
        row[f"{subj}__Grade"] = r["Grade"]
    for _, r in df_marks.iterrows():
        subj = r["Subject"]
        for comp, _mx in COMPONENTS:
            row[f"{subj}__{comp}"] = float(r[comp])
    for _, r in df_att.iterrows():
        subj = r["Subject"]
        row[f"{subj}__Held"] = int(r["Classes Held"])
        row[f"{subj}__Attended"] = int(r["Classes Attended"])
    for _, r in df_cred.iterrows():
        subj = r["Subject"]
        row[f"{subj}__Credits"] = float(r["Credits"])
    hist = load_history()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)
    return ts

def _user_dir():
    """Folder for the current user's saved state."""
    roll = st.session_state.get("roll_no")
    if not roll:
        return None
    d = os.path.join(DATA_DIR, f"user_{roll}")
    os.makedirs(d, exist_ok=True)
    return d

def save_current_state():
    """Save marks, attendance, credits, semester, and (optional) weights to CSV/TXT."""
    d = _user_dir()
    if not d:
        return
    try:
        st.session_state.marks.to_csv(os.path.join(d, "marks.csv"), index=False)
        st.session_state.attendance.to_csv(os.path.join(d, "attendance.csv"), index=False)
        st.session_state.credits.to_csv(os.path.join(d, "credits.csv"), index=False)
        with open(os.path.join(d, "semester.txt"), "w", encoding="utf-8") as f:
            f.write(str(st.session_state.get("semester", "")))
        # optional: exam weights if you added that feature
        if "component_max" in st.session_state:
            pd.DataFrame({
                "Component": list(st.session_state.component_max.keys()),
                "Max": [st.session_state.component_max[k] for k in st.session_state.component_max],
            }).to_csv(os.path.join(d, "weights.csv"), index=False)
    except Exception as e:
        st.warning(f"Could not save state: {e}")

def load_current_state():
    """Load marks, attendance, credits, semester, and (optional) weights if present."""
    d = _user_dir()
    if not d:
        return
    try:
        p_marks = os.path.join(d, "marks.csv")
        p_att   = os.path.join(d, "attendance.csv")
        p_cred  = os.path.join(d, "credits.csv")
        p_sem   = os.path.join(d, "semester.txt")
        p_wts   = os.path.join(d, "weights.csv")

        if os.path.exists(p_marks):
            st.session_state.marks = pd.read_csv(p_marks)
        if os.path.exists(p_att):
            st.session_state.attendance = pd.read_csv(p_att)
        if os.path.exists(p_cred):
            st.session_state.credits = pd.read_csv(p_cred)
        if os.path.exists(p_sem):
            try:
                st.session_state.semester = open(p_sem, "r", encoding="utf-8").read().strip()
            except:
                pass
        if os.path.exists(p_wts) and "component_max" in st.session_state:
            wdf = pd.read_csv(p_wts)
            st.session_state.component_max = {str(r["Component"]): float(r["Max"]) for _, r in wdf.iterrows()}
            # re-clamp using restored weights
            if "marks" in st.session_state:
                st.session_state.marks = clamp_marks(st.session_state.marks)
    except Exception as e:
        st.warning(f"Could not load previous state: {e}")

# ===================== SMOOTH CLIENT-SIDE ANIMATIONS (NO FLICKER) =====================
def _make_common_layout(title, y_max=None, shapes=None):
    layout = dict(
        template="plotly_dark",
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(tickangle=20),
        transition=dict(duration=300, easing="cubic-in-out"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 28, "redraw": True},
                    "transition": {"duration": 200, "easing": "cubic-in-out"},
                    "fromcurrent": True
                }]
            )]
        )],
        sliders=[dict(steps=[], transition={"duration": 0})]
    )
    if y_max is not None:
        layout["yaxis"] = dict(range=[0, y_max])
    if shapes:
        layout["shapes"] = shapes
    return layout

def _ease_cubic(t):
    return 4*t*t*t if t < 0.5 else 1 - (-2*t + 2)**3/2

def build_bar_animation(labels, start, end, title, y_max=None, pass_line=None, target_line=None, steps=60):
    start = np.asarray(start, dtype=float)
    end   = np.asarray(end, dtype=float)
    fig = go.Figure(data=[go.Bar(x=labels, y=start, marker_line_color="#22d3ee", marker_line_width=1)])
    shapes = []
    if pass_line is not None:
        shapes.append(dict(type="line", x0=-0.5, x1=len(labels)-0.5, y0=pass_line, y1=pass_line,
                           line=dict(dash="dash", color="#94a3b8")))
    if target_line is not None:
        shapes.append(dict(type="line", x0=-0.5, x1=len(labels)-0.5, y0=target_line, y1=target_line,
                           line=dict(dash="dash", color="#94a3b8")))
    frames = []
    for i in range(steps + 1):
        t = i / steps
        eased = _ease_cubic(t)
        y = start + (end - start) * eased
        frames.append(go.Frame(name=f"frame{i}", data=[go.Bar(x=labels, y=y)]))
    fig.frames = frames
    fig.update_layout(**_make_common_layout(title, y_max=y_max, shapes=shapes))
    fig["layout"]["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"frame{i}"], {
                 "mode": "immediate",
                 "frame": {"duration": 28, "redraw": True},
                 "transition": {"duration": 200, "easing": "cubic-in-out"}
             }],
             label=str(i))
        for i in range(len(fig.frames))
    ]
    return fig

def build_pie_animation(labels, start, end, title, steps=60):
    start = np.asarray(start, dtype=float); end = np.asarray(end, dtype=float)
    if float(np.sum(end)) <= 0:
        return go.Figure().update_layout(
            template="plotly_dark", title=title, height=420,
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)]
        )
    fig = go.Figure(data=[go.Pie(labels=labels, values=start, hole=0)])
    frames = []
    for i in range(steps + 1):
        t = i / steps
        eased = _ease_cubic(t)
        vals = start + (end - start) * eased
        frames.append(go.Frame(name=f"frame{i}", data=[go.Pie(labels=labels, values=vals, hole=0)]))
    fig.frames = frames
    fig.update_layout(**_make_common_layout(title))
    fig["layout"]["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"frame{i}"], {
                 "mode": "immediate",
                 "frame": {"duration": 28, "redraw": True},
                 "transition": {"duration": 200, "easing": "cubic-in-out"}
             }],
             label=str(i))
        for i in range(len(fig.frames))
    ]
    return fig

def build_line_animation(labels, start, end, title, y_max=None, steps=60):
    start = np.asarray(start, dtype=float); end = np.asarray(end, dtype=float)
    fig = go.Figure(data=[go.Scatter(x=labels, y=start, mode="lines+markers")])
    frames = []
    for i in range(steps + 1):
        t = i / steps
        eased = _ease_cubic(t)
        y = start + (end - start) * eased
        frames.append(go.Frame(name=f"frame{i}", data=[go.Scatter(x=labels, y=y, mode="lines+markers")]))
    fig.frames = frames
    fig.update_layout(**_make_common_layout(title, y_max=y_max))
    fig["layout"]["sliders"][0]["steps"] = [
        dict(method="animate",
             args=[[f"frame{i}"], {
                 "mode": "immediate",
                 "frame": {"duration": 28, "redraw": True},
                 "transition": {"duration": 200, "easing": "cubic-in-out"}
             }],
             label=str(i))
        for i in range(len(fig.frames))
    ]
    return fig

# ===================== HEADER / HERO =====================
st.markdown(
    """
    <div class="glass hero">
      <div class="pill">Student Analytics</div>
      <h1 style="margin:.35rem 0 0 0;">Smart Student Dashboard</h1>
      <p style="opacity:.85;">Animated • Dark • Secure — marks, attendance, SGPA/CGPA, snapshots & trends.</p>
      <div style="display:flex; gap:.5rem; flex-wrap:wrap;">
        <span class="chip">One-time access</span>
        <span class="chip">Hover pop-outs</span>
        <span class="chip">Dynamic charts</span>
        <span class="chip">History & trends</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["🏠 Home", "📚 Marks", "🕒 Attendance", "📈 History / CGPA", "⚙️ Settings"])

# ===================== AUTH GUARD =====================
def require_auth():
    if not st.session_state.authed or not st.session_state.roll_no:
        st.warning("Please unlock from **Home** to access this page.")
        st.stop()

def show_rollno():
    if st.session_state.roll_no:
        st.markdown(f"<div class='top-right'>Roll No: <b>{st.session_state.roll_no}</b></div>", unsafe_allow_html=True)

# ===================== HOME (Roll No + Password, once) =====================
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="glass" style="padding:18px;">', unsafe_allow_html=True)
        st.subheader("Welcome")
        st.markdown(
            "- Navigate via the tabs above.\n"
            "- Hover on cards for subtle pop-outs.\n"
            "- Charts run in dark mode with smooth transitions (no flicker).\n"
            "- Save **Snapshots** in History to build your timeline and CGPA."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass" style="padding:18px;">', unsafe_allow_html=True)
        st.subheader("Secure Access")
        if not st.session_state.authed:
            roll = st.text_input("Enter Roll Number", placeholder="e.g., 23CSE102")
            pwd = st.text_input("Enter Password", type="password", placeholder="••••••••")
            if st.button("Unlock", key="btn_unlock_home"):
                if pwd == PASSWORD and roll.strip():
                    st.session_state.authed = True
                    st.session_state.roll_no = roll.strip()
                    load_current_state()
                    st.success("Unlocked!")
                    st.rerun()
                else:
                    st.error("Invalid roll number or password")
        else:
            st.success("You’re already authenticated for this session.")
            st.caption("You can lock again from Settings if needed.")
            show_rollno()
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== MARKS =====================

with tabs[1]:
    require_auth()
    show_rollno()
    st.markdown('<div class="glass" style="padding:16px;">', unsafe_allow_html=True)
    st.markdown("#### 📚 Marks & Exams")

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        frames_smooth = st.slider("Animation smoothness (frames)", 20, 120, 60, help="Higher = smoother")
    with cB:
        chart_type = st.selectbox("Chart type", ["Bar", "Pie", "Line"])
    with cC:
        st.caption("Pass line at 40; Target line at 75")

   # Build dynamic column config based on current weights
    col_cfg = {}
    for comp, mx in st.session_state.component_max.items():
     col_cfg[comp] = st.column_config.NumberColumn(f"{comp} ({int(mx)})", min_value=0, max_value=int(mx), step=0.5)

    marks_df = st.data_editor(
    clamp_marks(st.session_state.marks),
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config=col_cfg,
    key="marks_editor"
)

    st.session_state.marks = clamp_marks(marks_df)
    save_current_state()


    totals = subject_totals(st.session_state.marks)
    df_tot = pd.DataFrame({"Subject": SUBJECTS, "Total": totals})
    df_tot["Grade"] = [grade_from_total(x) for x in totals]
    df_tot["Pass/Fail"] = np.where(df_tot["Total"] >= PASS_MARK, "Pass", "Fail")

    # ---- NEW: Performance Alerts ----
    threshold = st.session_state.perf_threshold
    below = df_tot[df_tot["Total"] < threshold][["Subject", "Total"]].sort_values("Total")

    if not below.empty:
     st.error(f"⚠️ Performance Alerts (below {threshold:.0f}):")
     st.dataframe(below, use_container_width=True)
    else:
     st.success(f"✅ All subjects are above {threshold:.0f}.")

    # KPIs
    def _k(vals):
        if len(vals) == 0: return (0, 0, 0, 0, 0)
        return (np.mean(vals), np.median(vals), np.max(vals), np.min(vals),
                np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    avg, med, best, worst, stdv = _k(totals)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Average", f"{avg:.1f}")
    m2.metric("Median", f"{med:.1f}")
    m3.metric("Best", f"{best:.0f}")
    m4.metric("Worst", f"{worst:.0f}")
    m5.metric("Std Dev", f"{stdv:.1f}")

    labels = df_tot["Subject"].tolist()
    start_vals = st.session_state.last_totals.copy()
    end_vals = totals.copy()

    if chart_type == "Bar":
        fig = build_bar_animation(labels, start_vals, end_vals, "Marks by Subject",
                                  y_max=100, pass_line=PASS_MARK, target_line=TARGET_MARK, steps=frames_smooth)
    elif chart_type == "Pie":
        fig = build_pie_animation(labels, start_vals, end_vals, "Share of Total Marks", steps=frames_smooth)
    else:
        fig = build_line_animation(labels, start_vals, end_vals, "Marks Trend", y_max=100, steps=frames_smooth)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="marks_chart")
    st.session_state.last_totals = end_vals.copy()
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== ATTENDANCE =====================
with tabs[2]:
    require_auth()
    show_rollno()
    st.markdown('<div class="glass" style="padding:16px;">', unsafe_allow_html=True)
    st.markdown("#### 🕒 Attendance Tracker")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        frames_att = st.slider("Animation smoothness (frames) ", 20, 120, 60, key="att_steps")
    with c2:
        att_chart = st.selectbox("Chart type (attendance)", ["Bar", "Pie", "Line"])
    with c3:
        st.caption("Minimum required: 75%")

    att_df = st.data_editor(
        st.session_state.attendance,
        hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={
            "Classes Held": st.column_config.NumberColumn("Classes Held", min_value=0, step=1),
            "Classes Attended": st.column_config.NumberColumn("Classes Attended", min_value=0, step=1),
        },
        key="att_editor"
    )
    att_df["Classes Held"] = att_df["Classes Held"].astype(int).clip(lower=0)
    att_df["Classes Attended"] = att_df["Classes Attended"].astype(int)
    att_df["Classes Attended"] = att_df[["Classes Attended", "Classes Held"]].min(axis=1).clip(lower=0)
    st.session_state.attendance = att_df
    save_current_state()


    out = att_df.copy()
    out["Attendance %"] = out.apply(lambda r: att_percent(r["Classes Held"], r["Classes Attended"]), axis=1)
    trip = out.apply(lambda r: next_class_advice(r["Classes Held"], r["Classes Attended"]), axis=1, result_type="expand")
    out["Current %"] = trip[0]; out["If Skip Next %"] = trip[1]; out["If Attend Next %"] = trip[2]; out["Advice"] = trip[3]

    labels = out["Subject"].tolist()
    start_vals = st.session_state.last_att.copy()
    end_vals = out["Attendance %"].to_numpy()

    if att_chart == "Bar":
        fig = build_bar_animation(labels, start_vals, end_vals, "Attendance by Subject",
                                  y_max=100, pass_line=ATT_REQ, target_line=None, steps=frames_att)
    elif att_chart == "Pie":
        fig = build_pie_animation(labels, start_vals, end_vals, "Attendance Share by Subject", steps=frames_att)
    else:
        fig = build_line_animation(labels, start_vals, end_vals, "Attendance Trend", y_max=100, steps=frames_att)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="att_chart")

    st.markdown("##### Recommendation per Subject")
    st.dataframe(out[["Subject", "Classes Held", "Classes Attended", "Current %", "If Skip Next %", "If Attend Next %", "Advice"]],
                 use_container_width=True)

    st.session_state.last_att = end_vals.copy()
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== HISTORY / CGPA =====================
with tabs[3]:
    require_auth()
    show_rollno()
    st.markdown('<div class="glass" style="padding:16px;">', unsafe_allow_html=True)
    st.markdown("#### 📈 History & CGPA")

    totals = subject_totals(st.session_state.marks)
    df_tot = pd.DataFrame({"Subject": SUBJECTS, "Total": totals})
    df_tot["Grade"] = [grade_from_total(x) for x in totals]

    sgpa, tot_cred = sgpa_calc(df_tot, st.session_state.credits)
    sgpa_2dp = sgpa.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)  # shows 8.02 when appropriate

    c1, c2, c3 = st.columns(3)
    c1.metric("Current SGPA", f"{sgpa_2dp}")
    c2.metric("Total Credits (semester)", f"{tot_cred:.1f}")
    hist_df = load_history()
    cgpa, cg_cred = cgpa_from_history(hist_df)
    cgpa_2dp = (cgpa if isinstance(cgpa, Decimal) else Decimal(str(cgpa))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    c3.metric("CGPA (latest per semester)", f"{cgpa_2dp}")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("💾 Save Snapshot",key="btn_save_snapshot"):
            ts = save_snapshot(st.session_state.semester, st.session_state.marks, st.session_state.attendance,
                               st.session_state.credits, df_tot, sgpa, tot_cred)
            st.toast(f"Snapshot saved at {ts}", icon="💾")

        # ========= PERSISTENCE: save/load current working state per Roll No =========
    def _user_dir():
     if not st.session_state.get("roll_no"):
        return None
     d = os.path.join(DATA_DIR, f"user_{st.session_state.roll_no}")
     os.makedirs(d, exist_ok=True)
     return d

    def save_current_state():
     """Save current marks, attendance, credits, semester, and (optionally) weights."""
     d = _user_dir()
     if not d:
      return
     try:
        st.session_state.marks.to_csv(os.path.join(d, "marks.csv"), index=False)
        st.session_state.attendance.to_csv(os.path.join(d, "attendance.csv"), index=False)
        st.session_state.credits.to_csv(os.path.join(d, "credits.csv"), index=False)
        # semester
        with open(os.path.join(d, "semester.txt"), "w", encoding="utf-8") as f:
            f.write(str(st.session_state.get("semester", "")))
        # optional: save exam weights if you added that feature
        if "component_max" in st.session_state:
            pd.DataFrame(
                {"Component": list(st.session_state.component_max.keys()),
                 "Max": [st.session_state.component_max[k] for k in st.session_state.component_max]}
            ).to_csv(os.path.join(d, "weights.csv"), index=False)
     except Exception as e:
        st.warning(f"Could not save state: {e}")

    def load_current_state():
     """Load previous state for this Roll No if present. Does NOT overwrite if files missing."""
     d = _user_dir()
     if not d:
        return
     try:
        p_marks = os.path.join(d, "marks.csv")
        p_att   = os.path.join(d, "attendance.csv")
        p_cred  = os.path.join(d, "credits.csv")
        p_sem   = os.path.join(d, "semester.txt")
        p_wts   = os.path.join(d, "weights.csv")

        if os.path.exists(p_marks):
            st.session_state.marks = pd.read_csv(p_marks)
        if os.path.exists(p_att):
            st.session_state.attendance = pd.read_csv(p_att)
        if os.path.exists(p_cred):
            st.session_state.credits = pd.read_csv(p_cred)
        if os.path.exists(p_sem):
            try:
                st.session_state.semester = open(p_sem, "r", encoding="utf-8").read().strip()
            except:
                pass
        # optional: restore weights
        if os.path.exists(p_wts) and "component_max" in st.session_state:
            wdf = pd.read_csv(p_wts)
            cm = {}
            for _, r in wdf.iterrows():
                cm[str(r["Component"])] = float(r["Max"])
            st.session_state.component_max = cm
            # re-clamp using restored weights
            if "marks" in st.session_state:
                st.session_state.marks = clamp_marks(st.session_state.marks)
     except Exception as e:
        st.warning(f"Could not load previous state: {e}")
    
    with colB:
        if os.path.exists(HISTORY_FILE):
            st.download_button("⬇️ Download History CSV", data=open(HISTORY_FILE, "rb").read(),
                               file_name="history.csv", mime="text/csv", key="btn_download_history")
    with colC:
        if not hist_df.empty:
            roll_hist = hist_df[hist_df.get("RollNo", "") == st.session_state.roll_no] if "RollNo" in hist_df.columns else hist_df
            options = roll_hist["Timestamp"].tolist()[::-1]
            pick = st.selectbox("Restore full snapshot:", options) if len(options) else None
            if pick and st.button("↩️ Restore", key="btn_restore_snapshot"):
                row = roll_hist[roll_hist["Timestamp"] == pick].iloc[0].to_dict()
                marks_rows = []
                for s in SUBJECTS:
                    rec = {"Subject": s}
                    for comp, _mx in COMPONENTS:
                        rec[comp] = float(row.get(f"{s}__{comp}", 0.0))
                    marks_rows.append(rec)
                st.session_state.marks = pd.DataFrame(marks_rows)
                att_rows = []
                for s in SUBJECTS:
                    att_rows.append({"Subject": s,
                                     "Classes Held": int(row.get(f"{s}__Held", 0)),
                                     "Classes Attended": int(row.get(f"{s}__Attended", 0))})
                st.session_state.attendance = pd.DataFrame(att_rows)
                cred_rows = []
                for s in SUBJECTS:
                    cred_rows.append({"Subject": s, "Credits": float(row.get(f"{s}__Credits", DEFAULT_CREDITS[s]))})
                st.session_state.credits = pd.DataFrame(cred_rows)
                st.toast("Snapshot restored.", icon="↩️")

    # SGPA over time (this roll only)
    hist_df = load_history()
    if not hist_df.empty:
        roll_hist = hist_df[hist_df.get("RollNo", "") == st.session_state.roll_no] if "RollNo" in hist_df.columns else hist_df
        if not roll_hist.empty:
            tdf = roll_hist.copy()
            tdf["Timestamp_dt"] = pd.to_datetime(tdf["Timestamp"])
            tdf = tdf.sort_values("Timestamp_dt")
            fig = go.Figure(go.Scatter(x=tdf["Timestamp_dt"], y=tdf["SGPA"], mode="lines+markers"))
            fig.update_layout(template="plotly_dark", title="SGPA Over Time", height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="sgpa_over_time")

            subj = st.selectbox("Per-subject trend", SUBJECTS)
            colname = f"{subj}__Total"
            if colname in roll_hist.columns:
                s2 = roll_hist[["Timestamp", colname]].dropna().copy()
                s2["Timestamp_dt"] = pd.to_datetime(s2["Timestamp"])
                s2 = s2.sort_values("Timestamp_dt")
                fig2 = go.Figure(go.Scatter(x=s2["Timestamp_dt"], y=s2[colname], mode="lines+markers"))
                fig2.update_layout(template="plotly_dark", title=f"{subj} — Total Marks Over Time",
                                   height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key="subject_trend")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== SETTINGS =====================
with tabs[4]:
    require_auth()
    show_rollno()
    st.markdown('<div class="glass" style="padding:16px;">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Settings")

    st.text_input("Semester label", key="semester")

    # ---- NEW: Threshold control ----
    st.session_state.perf_threshold = st.slider(
    "Performance alert threshold (total marks)",
    min_value=0, max_value=100, value=int(st.session_state.perf_threshold), step=1,
    help="Subjects below this total will be flagged in red on the Marks tab."
    )


    st.markdown("**Credits**")
    cred_edit = st.data_editor(
        st.session_state.credits,
        hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={"Credits": st.column_config.NumberColumn("Credits", min_value=0.0, step=0.5)},
        key="credits_editor"
    )
    cred_edit["Credits"] = cred_edit["Credits"].astype(float).clip(lower=0)
    st.session_state.credits = cred_edit
    save_current_state()

    
    # ---- Exam weightage (max marks) editor ----
st.markdown("**Exam weightage (max marks)**")

weights_df = pd.DataFrame(
    {
        "Component": list(st.session_state.component_max.keys()),
        "Max Marks": [st.session_state.component_max[k] for k in st.session_state.component_max],
    }
)

st.info("Adjust the max marks per exam component. The total should equal 100.", icon="ℹ️")

edited_weights = st.data_editor(
    weights_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Component": st.column_config.TextColumn("Component", disabled=True),
        "Max Marks": st.column_config.NumberColumn("Max Marks", min_value=0, max_value=200, step=1),
    },
    key="weights_editor"
)

# Validate and apply
try:
    new_map = {row["Component"]: float(row["Max Marks"]) for _, row in edited_weights.iterrows()}
    total_new = sum(new_map.values())
    if abs(total_new - 100.0) > 1e-6:
        st.warning(f"Total is {total_new:.0f}. For consistency keep it exactly 100.")
    else:
        if new_map != st.session_state.component_max:
            st.session_state.component_max = new_map
            st.session_state.marks = clamp_marks(st.session_state.marks)  # re-clamp to new maxima
            st.success("Exam weights updated and marks re-clamped.")
except Exception as e:
    st.error(f"Could not apply weights: {e}")

    


    st.write("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🔒 Lock (require password again)",key="btn_lock"):
            st.session_state.authed = False
            st.toast("Locked. Go to Home to unlock.", icon="🔒")
    with c2:
        if st.button("🧹 Reset to defaults",key="btn_reset_defaults"):
            st.session_state.marks = _default_marks_df()
            st.session_state.attendance = _default_att_df()
            st.session_state.credits = pd.DataFrame({"Subject": SUBJECTS, "Credits": [DEFAULT_CREDITS[s] for s in SUBJECTS]})
            st.toast("Reset done.", icon="🧹")
    with c3:
        if st.button("📦 Clear animation baselines",key="btn_clear_baselines"):
            st.session_state.last_totals = np.zeros(len(SUBJECTS))
            st.session_state.last_att = np.zeros(len(SUBJECTS))
            st.toast("Baselines cleared.", icon="📦")

    st.markdown("</div>", unsafe_allow_html=True)
