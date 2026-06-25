"""Data-access layer — the only module (besides auth) that touches the ORM.

Views call these functions with plain Python structures; nothing here imports
Streamlit, so the layer is testable against an in-memory SQLite database.

A "state" dict is the canonical shape passed around the UI::

    {
      "semester":   {"id", "label", "perf_threshold", "is_active"},
      "components": [{"id", "name", "max_marks", "order_index"}, ...],
      "subjects":   [{"id", "name", "credits", "order_index",
                      "scores": {component_name: value},
                      "attendance": {"held", "attended"}}, ...],
    }
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import academics
from .config import DEFAULT_COMPONENTS, DEFAULT_PERF_THRESHOLD, DEFAULT_SUBJECTS
from .db import session_scope
from .models import (
    Attendance,
    Component,
    Score,
    Semester,
    Snapshot,
    Subject,
    User,
)


# ---- internal ordered fetchers -----------------------------------------------
# These query the DB directly rather than reading ORM relationship collections,
# which can be stale within a session after rows are added/deleted by primary key.
def _subjects(session: Session, semester_id: int) -> list[Subject]:
    return list(session.scalars(
        select(Subject).where(Subject.semester_id == semester_id)
        .order_by(Subject.order_index, Subject.id)
    ))


def _components(session: Session, semester_id: int) -> list[Component]:
    return list(session.scalars(
        select(Component).where(Component.semester_id == semester_id)
        .order_by(Component.order_index, Component.id)
    ))


def _scores_for(session: Session, subject_ids: list[int]) -> list[Score]:
    if not subject_ids:
        return []
    return list(session.scalars(select(Score).where(Score.subject_id.in_(subject_ids))))


# ============================================================ users ===========
def get_user_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower().strip()))


def get_user_by_roll_no(session: Session, roll_no: str) -> User | None:
    """Look up a user by roll number (case-insensitive). Roll number is the
    login identifier, so this is how login resolves an account."""
    roll = (roll_no or "").strip()
    if not roll:
        return None
    return session.scalar(select(User).where(func.lower(User.roll_no) == roll.lower()))


def get_user(session: Session, user_id: int) -> User | None:
    return session.get(User, user_id)


def create_user(session: Session, *, email: str, password_hash: str,
                full_name: str = "", roll_no: str = "") -> User:
    user = User(
        email=email.lower().strip(),
        password_hash=password_hash,
        full_name=full_name.strip(),
        roll_no=roll_no.strip(),
    )
    session.add(user)
    session.flush()  # assign id
    seed_default_semester(session, user.id, label="Sem-1", make_active=True)
    return user


# ============================================================ semesters =======
def list_semesters(session: Session, user_id: int) -> list[Semester]:
    return list(
        session.scalars(
            select(Semester).where(Semester.user_id == user_id).order_by(Semester.created_at)
        )
    )


def get_active_semester(session: Session, user_id: int) -> Semester:
    sem = session.scalar(
        select(Semester).where(Semester.user_id == user_id, Semester.is_active.is_(True))
    )
    if sem is None:
        sem = session.scalar(
            select(Semester).where(Semester.user_id == user_id).order_by(Semester.created_at)
        )
        if sem is None:
            sem = seed_default_semester(session, user_id, make_active=True)
        else:
            sem.is_active = True
    return sem


def set_active_semester(session: Session, user_id: int, semester_id: int) -> None:
    for sem in list_semesters(session, user_id):
        sem.is_active = (sem.id == semester_id)


def seed_default_semester(session: Session, user_id: int, *, label: str = "Sem-1",
                          make_active: bool = False) -> Semester:
    """Create a semester pre-filled with the default subjects/components."""
    if make_active:
        for sem in list_semesters(session, user_id):
            sem.is_active = False
    sem = Semester(
        user_id=user_id, label=label, is_active=make_active,
        perf_threshold=DEFAULT_PERF_THRESHOLD,
    )
    session.add(sem)
    session.flush()

    for i, (cname, cmax) in enumerate(DEFAULT_COMPONENTS):
        session.add(Component(semester_id=sem.id, name=cname, max_marks=float(cmax), order_index=i))
    for i, (sname, credits) in enumerate(DEFAULT_SUBJECTS):
        subj = Subject(semester_id=sem.id, name=sname, credits=float(credits), order_index=i)
        session.add(subj)
        session.flush()
        session.add(Attendance(subject_id=subj.id, classes_held=0, classes_attended=0))
    session.flush()
    _ensure_score_grid(session, sem.id)
    return sem


def update_semester_meta(session: Session, semester_id: int, *, label: str | None = None,
                         perf_threshold: float | None = None) -> None:
    sem = session.get(Semester, semester_id)
    if sem is None:
        return
    if label is not None:
        sem.label = label.strip() or sem.label
    if perf_threshold is not None:
        sem.perf_threshold = float(perf_threshold)


def delete_semester(session: Session, user_id: int, semester_id: int) -> None:
    sems = list_semesters(session, user_id)
    if len(sems) <= 1:
        return  # never delete the last semester
    sem = session.get(Semester, semester_id)
    if sem is None or sem.user_id != user_id:
        return
    was_active = sem.is_active
    session.delete(sem)
    session.flush()
    if was_active:
        remaining = list_semesters(session, user_id)
        if remaining:
            remaining[0].is_active = True


# ============================================================ grid helpers ====
def _ensure_score_grid(session: Session, semester_id: int) -> None:
    """Guarantee exactly one Score row per (subject, component); clamp to max."""
    subjects = _subjects(session, semester_id)
    components = _components(session, semester_id)
    comp_max = {c.id: c.max_marks for c in components}
    valid_pairs = {(s.id, c.id) for s in subjects for c in components}

    existing = {
        (sc.subject_id, sc.component_id): sc
        for sc in _scores_for(session, [s.id for s in subjects])
    }
    # delete orphans (component/subject removed)
    for key, sc in existing.items():
        if key not in valid_pairs:
            session.delete(sc)
    # insert missing, clamp values
    for s in subjects:
        for c in components:
            sc = existing.get((s.id, c.id))
            if sc is None:
                session.add(Score(subject_id=s.id, component_id=c.id, value=0.0))
            else:
                sc.value = max(0.0, min(float(sc.value), comp_max.get(c.id, sc.value)))


# ============================================================ state read ======
def get_state(session: Session, semester_id: int) -> dict[str, Any]:
    sem = session.get(Semester, semester_id)
    if sem is None:
        raise ValueError(f"Semester {semester_id} not found")
    components = _components(session, semester_id)
    comp_by_id = {c.id: c for c in components}
    subjects = _subjects(session, semester_id)
    subject_ids = [s.id for s in subjects]

    # bulk-load scores + attendance to avoid per-subject queries
    scores_by_subject: dict[int, dict[int, float]] = {}
    for sc in _scores_for(session, subject_ids):
        scores_by_subject.setdefault(sc.subject_id, {})[sc.component_id] = sc.value
    att_by_subject = {
        a.subject_id: a
        for a in (session.scalars(
            select(Attendance).where(Attendance.subject_id.in_(subject_ids))
        ) if subject_ids else [])
    }

    state: dict[str, Any] = {
        "semester": {
            "id": sem.id, "label": sem.label,
            "perf_threshold": sem.perf_threshold, "is_active": sem.is_active,
            "sgpa_override": sem.sgpa_override,
        },
        "components": [
            {"id": c.id, "name": c.name, "max_marks": c.max_marks, "order_index": c.order_index}
            for c in components
        ],
        "subjects": [],
    }
    for s in subjects:
        by_comp = scores_by_subject.get(s.id, {})
        scores = {comp_by_id[cid].name: v for cid, v in by_comp.items() if cid in comp_by_id}
        att = att_by_subject.get(s.id)
        state["subjects"].append({
            "id": s.id, "name": s.name, "credits": s.credits, "order_index": s.order_index,
            "scores": {c.name: float(scores.get(c.name, 0.0)) for c in components},
            "attendance": {
                "held": int(att.classes_held) if att else 0,
                "attended": int(att.classes_attended) if att else 0,
            },
        })
    return state


# ============================================================ state writes ====
def save_scores(session: Session, semester_id: int,
                updates: dict[int, dict[str, float]]) -> None:
    """updates = {subject_id: {component_name: value}}; clamped to component max."""
    if session.get(Semester, semester_id) is None:
        return
    components = _components(session, semester_id)
    comp_by_name = {c.name: c for c in components}
    subject_ids = [s.id for s in _subjects(session, semester_id)]
    score_by_key = {
        (sc.subject_id, sc.component_id): sc
        for sc in _scores_for(session, subject_ids)
    }
    for subject_id, comp_values in updates.items():
        for cname, value in comp_values.items():
            comp = comp_by_name.get(cname)
            if comp is None:
                continue
            clamped = max(0.0, min(float(value), float(comp.max_marks)))
            sc = score_by_key.get((subject_id, comp.id))
            if sc is None:
                session.add(Score(subject_id=subject_id, component_id=comp.id, value=clamped))
            else:
                sc.value = clamped


def save_attendance(session: Session, updates: dict[int, tuple[int, int]]) -> None:
    """updates = {subject_id: (held, attended)}.

    Both are floored at 0. ``attended`` is capped at ``held`` only when classes
    have actually been held — capping against ``held == 0`` would silently wipe
    an "attended" value typed on a fresh row before its "held" was set (the
    attendance percentage already guards the ratio at display time).
    """
    for subject_id, (held, attended) in updates.items():
        held = max(0, int(held))
        attended = max(0, int(attended))
        if held > 0:
            attended = min(attended, held)
        att = session.scalar(select(Attendance).where(Attendance.subject_id == subject_id))
        if att is None:
            session.add(Attendance(subject_id=subject_id, classes_held=held,
                                   classes_attended=attended))
        else:
            att.classes_held = held
            att.classes_attended = attended


def set_subjects(session: Session, semester_id: int,
                 rows: list[dict[str, Any]]) -> None:
    """Reconcile the subject list. Each row: {id?, name, credits, order_index?}."""
    if session.get(Semester, semester_id) is None:
        return
    existing = {s.id: s for s in _subjects(session, semester_id)}
    keep_ids: set[int] = set()
    for i, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        credits = max(0.0, float(row.get("credits", 0) or 0))
        rid = row.get("id")
        if rid in existing:
            subj = existing[rid]
            subj.name, subj.credits, subj.order_index = name, credits, i
            keep_ids.add(rid)
        else:
            subj = Subject(semester_id=semester_id, name=name, credits=credits, order_index=i)
            session.add(subj)
            session.flush()
            session.add(Attendance(subject_id=subj.id, classes_held=0, classes_attended=0))
            keep_ids.add(subj.id)
    for sid, subj in existing.items():
        if sid not in keep_ids:
            session.delete(subj)
    session.flush()
    _ensure_score_grid(session, semester_id)


def rename_subjects(session: Session, semester_id: int, names: dict[int, str]) -> None:
    """Rename existing subjects by id (inline editing on the Marks tab).

    Unlike ``set_subjects`` this only touches the name — credits, scores and
    attendance are left untouched, and blank names are ignored.
    """
    if session.get(Semester, semester_id) is None:
        return
    by_id = {s.id: s for s in _subjects(session, semester_id)}
    for subject_id, new_name in names.items():
        subj = by_id.get(subject_id)
        clean = str(new_name).strip()
        if subj is not None and clean:
            subj.name = clean


def set_components(session: Session, semester_id: int,
                   rows: list[dict[str, Any]]) -> None:
    """Reconcile the component list. Each row: {id?, name, max_marks, order_index?}."""
    if session.get(Semester, semester_id) is None:
        return
    existing = {c.id: c for c in _components(session, semester_id)}
    keep_ids: set[int] = set()
    for i, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        cmax = max(0.0, float(row.get("max_marks", 0) or 0))
        rid = row.get("id")
        if rid in existing:
            comp = existing[rid]
            comp.name, comp.max_marks, comp.order_index = name, cmax, i
            keep_ids.add(rid)
        else:
            comp = Component(semester_id=semester_id, name=name, max_marks=cmax, order_index=i)
            session.add(comp)
            session.flush()
            keep_ids.add(comp.id)
    for cid, comp in existing.items():
        if cid not in keep_ids:
            session.delete(comp)
    session.flush()
    _ensure_score_grid(session, semester_id)


# ============================================================ metrics =========
def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Per-subject totals/grades + SGPA/total credits for a state dict."""
    comp_maxes = [c["max_marks"] for c in state["components"]]
    maximum = academics.max_total(comp_maxes)
    rows = []
    grade_credits = []
    for s in state["subjects"]:
        total = academics.subject_total(s["scores"].values())
        grade = academics.grade_for_total(total, maximum)
        rows.append({
            "Subject": s["name"], "Total": total, "Max": maximum,
            "Percent": academics.percent(total, maximum), "Grade": grade,
            "Result": "Pass" if academics.is_pass(total, maximum) else "Fail",
            "Credits": s["credits"],
        })
        grade_credits.append((grade, s["credits"]))
    sgpa_val, total_credits = academics.sgpa(grade_credits)
    # A manual SGPA override (if set on the semester) wins over the calculation.
    override = state["semester"].get("sgpa_override")
    if override is not None:
        sgpa_effective: Decimal = Decimal(str(override))
        sgpa_is_manual = True
    else:
        sgpa_effective = sgpa_val
        sgpa_is_manual = False
    return {
        "rows": rows, "sgpa": sgpa_val,
        "sgpa_effective": sgpa_effective, "sgpa_is_manual": sgpa_is_manual,
        "total_credits": total_credits, "max_total": maximum,
    }


# ============================================================ snapshots =======
def _state_to_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "semester_label": state["semester"]["label"],
        "perf_threshold": state["semester"]["perf_threshold"],
        "components": [
            {"name": c["name"], "max_marks": c["max_marks"], "order_index": c["order_index"]}
            for c in state["components"]
        ],
        "subjects": [
            {
                "name": s["name"], "credits": s["credits"], "order_index": s["order_index"],
                "scores": s["scores"],
                "attendance": s["attendance"],
            }
            for s in state["subjects"]
        ],
    }


def create_snapshot(session: Session, user_id: int, semester_id: int) -> Snapshot:
    state = get_state(session, semester_id)
    summary = summarize_state(state)
    snap = Snapshot(
        user_id=user_id,
        semester_id=semester_id,
        semester_label=state["semester"]["label"],
        taken_at=datetime.now(),
        sgpa=float(summary["sgpa_effective"]),  # honour a manual override
        total_credits=float(summary["total_credits"]),
        payload=_state_to_payload(state),
    )
    session.add(snap)
    session.flush()
    return snap


def list_snapshots(session: Session, user_id: int,
                   semester_id: int | None = None) -> list[Snapshot]:
    """Snapshots for a user, newest first.

    Pass ``semester_id`` to get only that semester's snapshots — each semester
    keeps its own independent history, so the History dropdown never mixes one
    semester's save points with another's.
    """
    stmt = select(Snapshot).where(Snapshot.user_id == user_id)
    if semester_id is not None:
        stmt = stmt.where(Snapshot.semester_id == semester_id)
    return list(session.scalars(stmt.order_by(Snapshot.taken_at.desc())))


def get_snapshot(session: Session, snapshot_id: int) -> Snapshot | None:
    return session.get(Snapshot, snapshot_id)


def restore_snapshot(session: Session, semester_id: int, snapshot_id: int) -> None:
    """Overwrite a semester's subjects/components/scores/attendance from one of
    *its own* snapshots.

    A snapshot can only be restored into the semester it was taken from, so one
    semester's subjects/marks can never bleed into another. Manual SGPA/CGPA
    overrides are independent, sticky settings and are deliberately left
    untouched here — restoring changes marks only.
    """
    snap = session.get(Snapshot, snapshot_id)
    sem = session.get(Semester, semester_id)
    if snap is None or sem is None:
        return
    if snap.semester_id is not None and snap.semester_id != semester_id:
        return  # refuse cross-semester restore — keeps Sem-1 out of Sem-4 etc.
    payload = snap.payload or {}
    update_semester_meta(session, semester_id,
                         perf_threshold=payload.get("perf_threshold"))
    set_components(session, semester_id, payload.get("components", []))
    set_subjects(session, semester_id, payload.get("subjects", []))
    # apply scores + attendance by subject name
    subj_by_name = {s.name: s for s in _subjects(session, semester_id)}
    score_updates: dict[int, dict[str, float]] = {}
    att_updates: dict[int, tuple[int, int]] = {}
    for s in payload.get("subjects", []):
        subj = subj_by_name.get(s["name"])
        if subj is None:
            continue
        score_updates[subj.id] = {k: float(v) for k, v in s.get("scores", {}).items()}
        att = s.get("attendance", {})
        att_updates[subj.id] = (int(att.get("held", 0)), int(att.get("attended", 0)))
    save_scores(session, semester_id, score_updates)
    save_attendance(session, att_updates)


def cgpa_auto_for_user(session: Session, user_id: int) -> tuple[Any, float]:
    """Credit-weighted CGPA from the latest snapshot per semester, *ignoring* any
    manual override → (value, total_credits). Used to show the auto value in
    Settings alongside a manual entry.
    """
    latest: dict[str, Snapshot] = {}
    for snap in list_snapshots(session, user_id):  # already newest-first
        latest.setdefault(snap.semester_label, snap)
    return academics.cgpa([(s.sgpa, s.total_credits) for s in latest.values()])


def cgpa_for_user(session: Session, user_id: int) -> tuple[Any, float, bool]:
    """CGPA for a user → (value, total_credits, is_manual).

    A manual override on the user wins; otherwise it is the credit-weighted
    average of the latest snapshot per semester label.
    """
    value, credits = cgpa_auto_for_user(session, user_id)
    user = session.get(User, user_id)
    if user is not None and user.cgpa_override is not None:
        return Decimal(str(user.cgpa_override)), credits, True
    return value, credits, False


def set_sgpa_override(session: Session, semester_id: int, value: float | None) -> None:
    """Set (or clear, with ``None``) a semester's manual SGPA."""
    sem = session.get(Semester, semester_id)
    if sem is not None:
        sem.sgpa_override = None if value is None else float(value)


def set_cgpa_override(session: Session, user_id: int, value: float | None) -> None:
    """Set (or clear, with ``None``) a user's manual CGPA."""
    user = session.get(User, user_id)
    if user is not None:
        user.cgpa_override = None if value is None else float(value)


# ---- convenience wrappers that open their own session (used by views) --------
def load_active_state(user_id: int) -> dict[str, Any]:
    with session_scope() as session:
        sem = get_active_semester(session, user_id)
        return get_state(session, sem.id)
