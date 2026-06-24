"""Unit tests for the pure academic calculations, including the two bug fixes."""
from __future__ import annotations

from decimal import Decimal

from ssd import academics


# ----------------------------------------------------------------- marks ------
def test_subject_total_and_percent():
    assert academics.subject_total([7, 21, 7, 35]) == 70
    assert academics.percent(70, 100) == 70.0
    # percent is 0 when the maximum is 0 (no divide-by-zero)
    assert academics.percent(50, 0) == 0.0


def test_grade_from_percent_bands():
    assert academics.grade_from_percent(95) == "A+"
    assert academics.grade_from_percent(85) == "A"
    assert academics.grade_from_percent(72) == "B+"
    assert academics.grade_from_percent(61) == "B"
    assert academics.grade_from_percent(50) == "C"
    assert academics.grade_from_percent(41) == "D"
    assert academics.grade_from_percent(39) == "F"


def test_grade_respects_custom_weightage_bug5():
    # Original bug: grading assumed a 100-mark scale. With a max of 50, a total
    # of 45 is 90% and must be A+, not "F" or "D".
    assert academics.grade_for_total(45, 50) == "A+"
    assert academics.grade_for_total(20, 50) == "D"   # 40%
    assert academics.grade_for_total(19, 50) == "F"   # 38%


def test_is_pass():
    assert academics.is_pass(40, 100) is True
    assert academics.is_pass(20, 50) is True   # 40%
    assert academics.is_pass(19, 50) is False


# -------------------------------------------------------------- attendance ----
def test_att_percent_zero_held_returns_zero_bug4():
    # Original bug: returned 100% when held==0 and attended>0.
    assert academics.att_percent(0, 5) == 0.0
    assert academics.att_percent(0, 0) == 0.0


def test_att_percent_clamps_attended_to_held():
    assert academics.att_percent(10, 12) == 100.0
    assert academics.att_percent(28, 22) == 100 * 22 / 28


def test_next_class_advice_messages():
    # Already comfortably above 75%: can skip.
    _, _, _, msg = academics.next_class_advice(20, 19)
    assert msg.startswith("✅")
    # Below but attending the next class recovers/holds.
    _, _, _, msg = academics.next_class_advice(3, 2)
    assert msg.startswith("⚠️") or msg.startswith("✅")
    # Far below and skipping makes it worse.
    _, _, _, msg = academics.next_class_advice(10, 2)
    assert msg.startswith("❌")


# ------------------------------------------------------------- sgpa / cgpa ----
def test_sgpa_credit_weighted():
    # A+ (10) x 4cr and C (6) x 2cr -> (40 + 12) / 6 = 8.667
    value, credits = academics.sgpa([("A+", 4), ("C", 2)])
    assert credits == 6.0
    assert value == Decimal("8.667")


def test_sgpa_zero_credits():
    value, credits = academics.sgpa([("A+", 0)])
    assert value == Decimal(0)
    assert credits == 0.0


def test_cgpa_credit_weighted():
    # Sem1 SGPA 8.0 x 20cr, Sem2 SGPA 9.0 x 20cr -> 8.5
    value, credits = academics.cgpa([(8.0, 20), (9.0, 20)])
    assert credits == 40.0
    assert value == Decimal("8.500")


def test_round_2dp():
    assert academics.round_2dp(Decimal("8.024")) == Decimal("8.02")
    assert academics.round_2dp_from_float(8.666) == Decimal("8.67")
