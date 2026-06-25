"""Integration tests for the repository layer against a real (SQLite) DB."""
from __future__ import annotations

from ssd import repository as repo
from ssd.config import DEFAULT_COMPONENTS, DEFAULT_SUBJECTS


def _make_user(session):
    user = repo.create_user(session, email="a@b.com", password_hash="x",
                            full_name="Test User", roll_no="R1")
    session.flush()
    return user


def test_create_user_seeds_default_semester(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)

    assert len(state["subjects"]) == len(DEFAULT_SUBJECTS)
    assert len(state["components"]) == len(DEFAULT_COMPONENTS)
    # every subject has a full score grid and an attendance record
    comp_names = {c["name"] for c in state["components"]}
    for subj in state["subjects"]:
        assert set(subj["scores"].keys()) == comp_names
        assert subj["attendance"] == {"held": 0, "attended": 0}


def test_save_scores_clamps_to_component_max(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)
    subj = state["subjects"][0]
    # "Sessional 1" max is 10 — try to store 999, expect clamp to 10.
    repo.save_scores(session, sem.id, {subj["id"]: {"Sessional 1": 999}})
    state2 = repo.get_state(session, sem.id)
    assert state2["subjects"][0]["scores"]["Sessional 1"] == 10.0


def test_save_attendance_clamps_attended_to_held(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)
    subj = state["subjects"][0]
    repo.save_attendance(session, {subj["id"]: (10, 25)})
    state2 = repo.get_state(session, sem.id)
    assert state2["subjects"][0]["attendance"] == {"held": 10, "attended": 10}


def test_save_attendance_keeps_attended_when_held_zero(session):
    # Bug: entering "attended" on a fresh row (held=0) used to clamp it to 0,
    # making the edit vanish. It must now survive until "held" is set.
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    subj = repo.get_state(session, sem.id)["subjects"][0]
    repo.save_attendance(session, {subj["id"]: (0, 18)})
    att = repo.get_state(session, sem.id)["subjects"][0]["attendance"]
    assert att == {"held": 0, "attended": 18}
    # once held is set, the normal cap applies again
    repo.save_attendance(session, {subj["id"]: (20, 18)})
    att = repo.get_state(session, sem.id)["subjects"][0]["attendance"]
    assert att == {"held": 20, "attended": 18}


def test_set_subjects_add_and_remove(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    # replace with two subjects only
    repo.set_subjects(session, sem.id, [
        {"id": None, "name": "Alpha", "credits": 3},
        {"id": None, "name": "Beta", "credits": 4},
    ])
    state = repo.get_state(session, sem.id)
    assert [s["name"] for s in state["subjects"]] == ["Alpha", "Beta"]
    # score grid rebuilt for the new subjects
    comp_names = {c["name"] for c in state["components"]}
    for subj in state["subjects"]:
        assert set(subj["scores"].keys()) == comp_names


def test_rename_subjects_keeps_credits_and_scores(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)
    subj = state["subjects"][0]
    repo.save_scores(session, sem.id, {subj["id"]: {"Sessional 1": 7}})

    # rename trims whitespace; blank names and unknown ids are ignored
    repo.rename_subjects(session, sem.id, {subj["id"]: "  Renamed  ", 999999: "ghost"})
    repo.rename_subjects(session, sem.id, {subj["id"]: "   "})

    state2 = repo.get_state(session, sem.id)
    renamed = next(s for s in state2["subjects"] if s["id"] == subj["id"])
    assert renamed["name"] == "Renamed"
    assert renamed["credits"] == subj["credits"]
    assert renamed["scores"]["Sessional 1"] == 7.0


def test_set_components_reclamps_scores(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)
    subj = state["subjects"][0]
    repo.save_scores(session, sem.id, {subj["id"]: {"End Term": 50}})

    # Shrink End Term max from 50 to 20 — stored 50 must re-clamp to 20.
    comps = state["components"]
    new_rows = []
    for c in comps:
        new_rows.append({"id": c["id"], "name": c["name"],
                         "max_marks": 20 if c["name"] == "End Term" else c["max_marks"]})
    repo.set_components(session, sem.id, new_rows)
    state2 = repo.get_state(session, sem.id)
    assert state2["subjects"][0]["scores"]["End Term"] == 20.0


def test_snapshot_restore_and_cgpa(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    state = repo.get_state(session, sem.id)
    # give the first subject full marks, then snapshot
    sid0 = state["subjects"][0]["id"]
    repo.save_scores(session, sem.id, {
        sid0: {"Sessional 1": 10, "Mid Term": 30, "Sessional 2": 10, "End Term": 50}
    })
    snap = repo.create_snapshot(session, user.id, sem.id)
    assert snap.total_credits > 0

    snaps = repo.list_snapshots(session, user.id)
    assert len(snaps) == 1

    cgpa_val, credits, manual = repo.cgpa_for_user(session, user.id)
    assert credits > 0
    assert float(cgpa_val) > 0
    assert manual is False

    # wipe the score, then restore the snapshot and confirm it came back
    repo.save_scores(session, sem.id, {sid0: {"Sessional 1": 0, "Mid Term": 0,
                                              "Sessional 2": 0, "End Term": 0}})
    repo.restore_snapshot(session, sem.id, snap.id)
    restored = repo.get_state(session, sem.id)
    first = next(s for s in restored["subjects"] if s["id"] == sid0 or s["name"] ==
                 state["subjects"][0]["name"])
    assert first["scores"]["End Term"] == 50.0


def test_sgpa_override(session):
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)

    summ = repo.summarize_state(repo.get_state(session, sem.id))
    assert summ["sgpa_is_manual"] is False  # auto by default

    repo.set_sgpa_override(session, sem.id, 9.5)
    summ = repo.summarize_state(repo.get_state(session, sem.id))
    assert summ["sgpa_is_manual"] is True
    assert float(summ["sgpa_effective"]) == 9.5
    # a snapshot records the override value (so CGPA reflects it)
    snap = repo.create_snapshot(session, user.id, sem.id)
    assert float(snap.sgpa) == 9.5

    repo.set_sgpa_override(session, sem.id, None)  # clear -> back to auto
    summ = repo.summarize_state(repo.get_state(session, sem.id))
    assert summ["sgpa_is_manual"] is False


def test_restore_leaves_manual_override_untouched(session):
    """A manual SGPA override is an independent, sticky setting: restore changes
    marks only and must neither set nor clear it."""
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    sid0 = repo.get_state(session, sem.id)["subjects"][0]["id"]

    first_name = repo.get_state(session, sem.id)["subjects"][0]["name"]
    repo.set_sgpa_override(session, sem.id, 8.5)
    repo.save_scores(session, sem.id, {sid0: {"End Term": 50}})
    snap = repo.create_snapshot(session, user.id, sem.id)
    repo.save_scores(session, sem.id, {sid0: {"End Term": 10}})  # diverge

    repo.restore_snapshot(session, sem.id, snap.id)
    summ = repo.summarize_state(repo.get_state(session, sem.id))
    assert summ["sgpa_is_manual"] is True            # override untouched
    assert float(summ["sgpa_effective"]) == 8.5
    restored = repo.get_state(session, sem.id)
    first = next(s for s in restored["subjects"] if s["name"] == first_name)
    assert first["scores"]["End Term"] == 50.0       # marks were restored


def test_restore_auto_snapshot_does_not_add_override(session):
    """Restoring a snapshot never introduces an override on a semester that had
    none, so the SGPA stays auto/live."""
    user = _make_user(session)
    sem = repo.get_active_semester(session, user.id)
    sid0 = repo.get_state(session, sem.id)["subjects"][0]["id"]

    repo.save_scores(session, sem.id, {sid0: {"End Term": 50, "Mid Term": 30}})
    snap = repo.create_snapshot(session, user.id, sem.id)
    repo.save_scores(session, sem.id, {sid0: {"End Term": 0, "Mid Term": 0}})
    repo.restore_snapshot(session, sem.id, snap.id)
    summ = repo.summarize_state(repo.get_state(session, sem.id))
    assert summ["sgpa_is_manual"] is False


def test_snapshots_scoped_per_semester(session):
    """Each semester keeps its own snapshots; one semester's snapshot can never
    be restored into another."""
    user = _make_user(session)
    sem1 = repo.get_active_semester(session, user.id)
    sem2 = repo.seed_default_semester(session, user.id, label="Sem-2", make_active=True)

    s1_subj = repo.get_state(session, sem1.id)["subjects"][0]["id"]
    repo.save_scores(session, sem1.id, {s1_subj: {"End Term": 40}})
    snap1 = repo.create_snapshot(session, user.id, sem1.id)
    snap2 = repo.create_snapshot(session, user.id, sem2.id)

    # the dropdown for each semester only sees its own snapshots
    assert {s.id for s in repo.list_snapshots(session, user.id, sem1.id)} == {snap1.id}
    assert {s.id for s in repo.list_snapshots(session, user.id, sem2.id)} == {snap2.id}

    # restoring Sem-1's snapshot into Sem-2 is refused — no cross-semester bleed
    before = [s["name"] for s in repo.get_state(session, sem2.id)["subjects"]]
    repo.restore_snapshot(session, sem2.id, snap1.id)
    after = [s["name"] for s in repo.get_state(session, sem2.id)["subjects"]]
    assert before == after


def test_cgpa_override(session):
    user = _make_user(session)
    repo.set_cgpa_override(session, user.id, 8.25)
    value, _credits, manual = repo.cgpa_for_user(session, user.id)
    assert manual is True
    assert float(value) == 8.25

    repo.set_cgpa_override(session, user.id, None)
    _value, _credits, manual = repo.cgpa_for_user(session, user.id)
    assert manual is False


def test_ensure_schema_adds_missing_columns():
    import os
    import tempfile

    from sqlalchemy import create_engine, inspect, text

    from ssd import db

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_engine(f"sqlite:///{path}")
    # simulate an older database without the override columns
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE semesters (id INTEGER PRIMARY KEY, label TEXT)"))
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)"))

    db.ensure_schema(eng)
    db.ensure_schema(eng)  # idempotent — second run must not error

    sem_cols = {c["name"] for c in inspect(eng).get_columns("semesters")}
    user_cols = {c["name"] for c in inspect(eng).get_columns("users")}
    assert "sgpa_override" in sem_cols
    assert "cgpa_override" in user_cols

    eng.dispose()
    os.remove(path)
