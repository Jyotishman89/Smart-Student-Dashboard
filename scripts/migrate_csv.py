"""One-time migration: legacy CSV files -> the new database.

Imports each ``user_<roll>/`` working-state folder as a user with an active
semester, and imports ``history.csv`` rows as snapshots. Idempotent: users that
already exist (matched by email) are skipped.

Usage:
    python scripts/migrate_csv.py --data-dir "<path to student_data>" --dry-run
    python scripts/migrate_csv.py --data-dir "<path to student_data>"

If --data-dir is omitted it defaults to the known OneDrive location.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ssd import auth  # noqa: E402
from ssd import repository as repo
from ssd.config import DEFAULT_COMPONENTS  # noqa: E402
from ssd.db import init_db, session_scope  # noqa: E402
from ssd.models import Snapshot  # noqa: E402

DEFAULT_DATA_DIR = (
    Path.home() / "OneDrive" / "Desktop" / "Python"
    / "Student Smart Dashboard" / "student_data"
)
PLACEHOLDER_PASSWORD = "ChangeMe123!"  # users must reset after migration


# ------------------------------------------------------------ folder import ---
def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def import_user_folder(session, folder: Path, *, dry_run: bool) -> str:
    roll = folder.name.replace("user_", "", 1)
    email = f"{roll.lower()}@migrated.local"

    if repo.get_user_by_email(session, email):
        return f"skip  {roll} (already migrated)"
    if dry_run:
        return f"would create user {roll} <{email}> from {folder.name}"

    marks = _read_csv(folder / "marks.csv")
    attendance = _read_csv(folder / "attendance.csv")
    credits = _read_csv(folder / "credits.csv")
    weights = _read_csv(folder / "weights.csv")
    sem_file = folder / "semester.txt"
    label = sem_file.read_text(encoding="utf-8").strip() if sem_file.exists() else ""
    label = label or "Sem-1"

    user = repo.create_user(
        session, email=email, password_hash=auth.hash_password(PLACEHOLDER_PASSWORD),
        full_name=roll, roll_no=roll,
    )
    sem = repo.get_active_semester(session, user.id)
    repo.update_semester_meta(session, sem.id, label=label)

    # components (from weights.csv, else defaults)
    if weights is not None:
        comp_rows = [{"id": None, "name": str(r["Component"]), "max_marks": float(r["Max"])}
                     for _, r in weights.iterrows()]
    else:
        comp_rows = [{"id": None, "name": n, "max_marks": float(m)} for n, m in DEFAULT_COMPONENTS]
    repo.set_components(session, sem.id, comp_rows)

    # subjects (+ credits merged by name)
    credit_map = {}
    if credits is not None:
        credit_map = {str(r["Subject"]): float(r["Credits"]) for _, r in credits.iterrows()}
    subj_rows = []
    if marks is not None:
        for _, r in marks.iterrows():
            name = str(r["Subject"])
            subj_rows.append({"id": None, "name": name, "credits": credit_map.get(name, 0.0)})
    repo.set_subjects(session, sem.id, subj_rows)

    # scores + attendance, mapped by subject name
    state = repo.get_state(session, sem.id)
    id_by_name = {s["name"]: s["id"] for s in state["subjects"]}
    comp_names = [c["name"] for c in state["components"]]

    if marks is not None:
        score_updates = {}
        for _, r in marks.iterrows():
            sid = id_by_name.get(str(r["Subject"]))
            if sid is None:
                continue
            score_updates[sid] = {cn: float(r[cn]) for cn in comp_names if cn in marks.columns}
        repo.save_scores(session, sem.id, score_updates)

    if attendance is not None:
        att_updates = {}
        for _, r in attendance.iterrows():
            sid = id_by_name.get(str(r["Subject"]))
            if sid is None:
                continue
            att_updates[sid] = (int(r["Classes Held"]), int(r["Classes Attended"]))
        repo.save_attendance(session, att_updates)

    return f"create {roll}: {len(subj_rows)} subjects, {len(comp_rows)} components"


# ----------------------------------------------------------- history import ---
def _clean_label(value) -> str:
    """Coerce an empty / NaN Semester field to a sensible default.

    Note: a float NaN is *truthy*, so ``nan or "Sem-1"`` returns nan — hence the
    explicit pd.isna check rather than relying on ``or``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Sem-1"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "Sem-1"


def _history_subjects(columns: list[str]) -> list[str]:
    return sorted({c.split("__")[0] for c in columns if c.endswith("__Total")})


def import_history(session, history_path: Path, *, dry_run: bool) -> list[str]:
    if not history_path.exists():
        return ["no history.csv found"]
    hist = pd.read_csv(history_path)
    if hist.empty:
        return ["history.csv is empty"]

    subjects = _history_subjects(list(hist.columns))
    comp_names = [n for n, _ in DEFAULT_COMPONENTS]
    has_roll = "RollNo" in hist.columns

    # Map roll -> user id for attribution; fall back to the first migrated user.
    msgs: list[str] = []
    fallback_uid = None
    first_user = None
    from ssd.models import User
    first_user = session.query(User).order_by(User.id).first()
    if first_user:
        fallback_uid = first_user.id

    created = skipped = 0
    for _, row in hist.iterrows():
        uid = fallback_uid
        if has_roll and str(row.get("RollNo", "")).strip():
            u = repo.get_user_by_email(session, f"{str(row['RollNo']).lower()}@migrated.local")
            uid = u.id if u else fallback_uid
        if uid is None:
            continue
        payload = _row_to_payload(row, subjects, comp_names)
        if dry_run:
            continue
        ts = pd.to_datetime(row["Timestamp"]) if "Timestamp" in row else datetime.now()
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        # idempotency: skip if this user already has a snapshot at this timestamp
        if session.scalar(select(Snapshot).where(
            Snapshot.user_id == uid, Snapshot.taken_at == ts
        )):
            skipped += 1
            continue
        session.add(Snapshot(
            user_id=uid, semester_id=None,
            semester_label=_clean_label(row.get("Semester")),
            taken_at=ts,
            sgpa=float(row.get("SGPA", 0) or 0),
            total_credits=float(row.get("TotalCredits", 0) or 0),
            payload=payload,
        ))
        created += 1

    verb = "would import" if dry_run else "imported"
    msgs.append(f"{verb} {len(hist)} history rows as snapshots "
                f"(attributed by {'RollNo' if has_roll else 'first user'})")
    if not dry_run:
        msgs.append(f"created {created} snapshots, skipped {skipped} duplicates")
    return msgs


def _row_to_payload(row, subjects: list[str], comp_names: list[str]) -> dict:
    comps = [{"name": n, "max_marks": float(m), "order_index": i}
             for i, (n, m) in enumerate(DEFAULT_COMPONENTS)]
    subj_payload = []
    for i, name in enumerate(subjects):
        scores = {cn: float(row.get(f"{name}__{cn}", 0) or 0) for cn in comp_names}
        held = int(row.get(f"{name}__Held", 0) or 0)
        attended = int(row.get(f"{name}__Attended", 0) or 0)
        credits = float(row.get(f"{name}__Credits", 0) or 0)
        subj_payload.append({
            "name": name, "credits": credits, "order_index": i,
            "scores": scores, "attendance": {"held": held, "attended": attended},
        })
    return {"semester_label": _clean_label(row.get("Semester")),
            "perf_threshold": 50.0, "components": comps, "subjects": subj_payload}


# ------------------------------------------------------------------- main -----
def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy CSV data into the database.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, no writes.")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    print(f"Data dir: {data_dir}")
    if not data_dir.exists():
        print("  ! data dir does not exist — nothing to migrate.")
        return

    if not args.dry_run:
        init_db()

    with session_scope() as session:
        for folder in sorted(data_dir.glob("user_*")):
            if folder.is_dir():
                print("  " + import_user_folder(session, folder, dry_run=args.dry_run))
        for msg in import_history(session, data_dir / "history.csv", dry_run=args.dry_run):
            print("  " + msg)

    print("Dry run complete." if args.dry_run else "Migration complete.")


if __name__ == "__main__":
    main()
