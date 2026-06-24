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

    cgpa_val, credits = repo.cgpa_for_user(session, user.id)
    assert credits > 0
    assert float(cgpa_val) > 0

    # wipe the score, then restore the snapshot and confirm it came back
    repo.save_scores(session, sem.id, {sid0: {"Sessional 1": 0, "Mid Term": 0,
                                              "Sessional 2": 0, "End Term": 0}})
    repo.restore_snapshot(session, sem.id, snap.id)
    restored = repo.get_state(session, sem.id)
    first = next(s for s in restored["subjects"] if s["id"] == sid0 or s["name"] ==
                 state["subjects"][0]["name"])
    assert first["scores"]["End Term"] == 50.0
