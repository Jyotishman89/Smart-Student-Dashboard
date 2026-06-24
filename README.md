# Smart Student Dashboard

A **multi-user academic dashboard** built with [Streamlit](https://streamlit.io/) and
PostgreSQL. Track marks, attendance, SGPA and CGPA — securely, across semesters.

> Rebuilt from a single-file prototype into a modular, tested, database-backed app.

🔗 **Live:** https://jyotishman89-smart-student-dashboard-ssd-r8odh3.streamlit.app/

---

## Features

- **Real accounts** — sign up with your name, **roll number** and email; then **log in with your roll number + password** (hashed with **bcrypt**). Each user's data is private.
- **Cloud database** — PostgreSQL (Neon). Data persists across redeploys.
- **Marks & grades** — editable grid, live grade/percentage, pass/fail and performance alerts.
- **Attendance** — per-subject tracking with skip-vs-attend advice for the next class.
- **SGPA / CGPA** — credit-weighted, exact decimal rounding; **snapshots** build your CGPA timeline.
- **Fully customisable** — add/remove subjects, set credits, and define your own exam weightage per semester.
- Dark, glassmorphism UI with clean Plotly charts.

---

## Architecture

```
ssd.py                 # thin Streamlit entrypoint -> ssd.main.run()
ssd/
  config.py            # constants + settings (secrets/env)
  db.py                # SQLAlchemy engine (Postgres; SQLite fallback for dev/tests)
  models.py            # ORM: User, Semester, Subject, Component, Score, Attendance, Snapshot
  academics.py         # PURE calc logic (grades, SGPA, CGPA, attendance) — unit tested
  repository.py        # data-access (CRUD)
  auth.py              # bcrypt signup/login + cookie sessions
  charts.py            # Plotly figure builders
  theme.py             # page config + CSS
  main.py              # auth gate + st.navigation routing
  views/               # one module per page (home, marks, attendance, history, settings)
scripts/
  init_db.py           # create tables (idempotent)
  migrate_csv.py       # one-time legacy CSV -> DB import
tests/                 # pytest (academics + repository)
```

The calculation layer (`academics.py`) has **no Streamlit or DB imports**, which is what makes it fully unit-testable.

---

## Local setup

```bash
git clone https://github.com/Jyotishman89/Smart-Student-Dashboard.git
cd Smart-Student-Dashboard

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Configure secrets

Copy the example and fill it in:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
# .streamlit/secrets.toml  (gitignored)
DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DB?sslmode=require"   # Neon pooled string
COOKIE_SECRET = "a-long-random-string"
```

> **No DATABASE_URL?** The app automatically falls back to a local SQLite file
> (`app.db`) so you can run everything offline.

### Run

```bash
python scripts/init_db.py     # create tables
streamlit run ssd.py
```

---

## Tests & lint

```bash
pytest -q          # unit + repository tests
ruff check .       # lint
```

CI runs both on every push/PR (`.github/workflows/ci.yml`).

---

## Migrating old CSV data

```bash
python scripts/migrate_csv.py --dry-run     # preview
python scripts/migrate_csv.py               # import users + history snapshots
```

Migrated accounts use a placeholder password (`ChangeMe123!`) — reset it after first login.

---

## ☁️ Deploy (Streamlit Community Cloud)

1. Push to GitHub.
2. In the app's **Settings → Secrets**, add `DATABASE_URL` and `COOKIE_SECRET`.
3. Main file path stays **`ssd.py`**.

---

## Future work

- Email verification & password reset
- Per-user theme preferences
- Export to PDF report card

---

## 📜 Disclaimer

Built for learning. Passwords are bcrypt-hashed and data is per-user isolated, but this is a
student project — review security before using it with sensitive data at scale.
