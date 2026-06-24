"""Pure academic calculations — no Streamlit, no database, fully unit-testable.

This module is the single source of truth for marks totals, grading, attendance
advice and SGPA/CGPA. Keeping it side-effect free is what makes it testable and
is where the original script's two arithmetic bugs are fixed:

* attendance now returns 0%% when no classes have been held (was 100%%);
* grades are computed from *percentage of maximum*, so they stay correct when a
  user customises component weightage (the original hard-coded a 100-mark scale).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal, getcontext

from .config import (
    ATTENDANCE_REQ,
    GRADE_BANDS,
    GRADE_POINTS,
    PASS_PERCENT,
)

getcontext().prec = 28


# --------------------------------------------------------------------- marks --
def subject_total(component_values: Iterable[float]) -> float:
    """Sum the component scores for one subject."""
    return float(sum(float(v) for v in component_values))


def max_total(component_maxes: Iterable[float]) -> float:
    """Sum the maximum marks across components (the per-subject denominator)."""
    return float(sum(float(v) for v in component_maxes))


def percent(total: float, maximum: float) -> float:
    """Total as a percentage of the maximum. 0 when maximum is 0."""
    return (100.0 * float(total) / float(maximum)) if maximum > 0 else 0.0


def grade_from_percent(pct: float) -> str:
    """Letter grade from a percentage using the configured bands."""
    for cutoff, grade in GRADE_BANDS:
        if pct >= cutoff:
            return grade
    return "F"


def grade_for_total(total: float, maximum: float) -> str:
    """Letter grade for a raw total given its maximum (handles custom weightage)."""
    return grade_from_percent(percent(total, maximum))


def is_pass(total: float, maximum: float) -> bool:
    return percent(total, maximum) >= PASS_PERCENT


# ---------------------------------------------------------------- attendance --
def att_percent(held: int, attended: int) -> float:
    """Attendance percentage. Returns 0 when no classes were held (bug fix)."""
    held = max(0, int(held))
    attended = max(0, min(int(attended), held))
    return (100.0 * attended / held) if held > 0 else 0.0


def next_class_advice(held: int, attended: int, req: float = ATTENDANCE_REQ):
    """Return (current%, if-skip%, if-attend%, advice message)."""
    cur = att_percent(held, attended)
    skip_p = att_percent(held + 1, attended)
    attend_p = att_percent(held + 1, attended + 1)
    if skip_p >= req:
        msg = "✅ You can skip the next class and stay above the limit."
    elif attend_p >= req:
        msg = "⚠️ Attend the next class to stay/return above the limit."
    else:
        msg = "❌ Attend the next class — skipping makes it worse."
    return round(cur, 2), round(skip_p, 2), round(attend_p, 2), msg


# ----------------------------------------------------------------- sgpa/cgpa --
def _q(value: Decimal, places: str = "0.001") -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def sgpa(grade_credits: Sequence[tuple[str, float]]) -> tuple[Decimal, float]:
    """Credit-weighted SGPA.

    Args:
        grade_credits: sequence of (letter_grade, credits).
    Returns:
        (sgpa rounded to 3dp, total credits).
    """
    weighted = Decimal(0)
    total_credits = Decimal(0)
    for grade, credits in grade_credits:
        gp = Decimal(int(GRADE_POINTS.get(grade, 0)))
        cr = Decimal(str(credits))
        weighted += gp * cr
        total_credits += cr
    value = Decimal(0) if total_credits == 0 else _q(weighted / total_credits)
    return value, float(total_credits)


def cgpa(semester_results: Sequence[tuple[float, float]]) -> tuple[Decimal, float]:
    """Credit-weighted CGPA across semesters.

    Args:
        semester_results: sequence of (sgpa, total_credits) — one per semester.
    Returns:
        (cgpa rounded to 3dp, total credits across all semesters).
    """
    weighted = Decimal(0)
    total_credits = Decimal(0)
    for sg, credits in semester_results:
        s = Decimal(str(sg))
        cr = Decimal(str(credits))
        weighted += s * cr
        total_credits += cr
    value = Decimal(0) if total_credits == 0 else _q(weighted / total_credits)
    return value, float(total_credits)


def round_2dp(value: Decimal) -> Decimal:
    """Display helper — round a 3dp gpa to 2dp (e.g. 8.024 -> 8.02)."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def round_2dp_from_float(value: float) -> Decimal:
    """Same as round_2dp but accepts a plain float (e.g. a stored SGPA)."""
    return round_2dp(Decimal(str(value)))
