"""Behavioral availability modifier from redrob_signals. Returns a multiplier
in [BEHAVIOR_FLOOR, 1.0] that down-weights candidates who aren't realistically
hireable (inactive, unresponsive, not open)."""
from datetime import date
from typing import Dict, Any
from src import config as C

def _days_inactive(last_active: str) -> int:
    try:
        y, m, d = (int(x) for x in last_active.split("-"))
        return (C.REFERENCE_DATE - date(y, m, d)).days
    except Exception:
        return 365

def behavior_modifier(c: Dict[str, Any]) -> float:
    s = c.get("redrob_signals", {})
    score = 1.0

    days = _days_inactive(s.get("last_active_date", ""))
    if days > 180:
        score -= 0.25
    elif days > 90:
        score -= 0.12
    elif days > 45:
        score -= 0.05

    rr = float(s.get("recruiter_response_rate", 0.5) or 0.0)
    if rr < 0.1:
        score -= 0.20
    elif rr < 0.3:
        score -= 0.10

    if not s.get("open_to_work_flag", False):
        score -= 0.08

    icr = s.get("interview_completion_rate", None)
    if icr is not None and float(icr) < 0.5:
        score -= 0.05

    return max(C.BEHAVIOR_FLOOR, min(1.0, score))
