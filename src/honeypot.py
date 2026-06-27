"""Detect subtly impossible profiles ('honeypots'). Any candidate tripping a
check is later demoted below the top-100 cutoff. Conservative thresholds so we
never flag a legitimate candidate."""
from datetime import date
from typing import Dict, Any
from src import config as C

def _elapsed_months(start: str, end: str | None) -> float:
    try:
        ys, ms, ds = (int(x) for x in start.split("-"))
        start_d = date(ys, ms, ds)
    except Exception:
        return 1e9  # unparseable start -> don't use this check
    if end:
        try:
            ye, me, de = (int(x) for x in end.split("-"))
            end_d = date(ye, me, de)
        except Exception:
            end_d = C.REFERENCE_DATE
    else:
        end_d = C.REFERENCE_DATE
    return max(0.0, (end_d - start_d).days / 30.44)

def is_honeypot(c: Dict[str, Any]) -> bool:
    ch = c.get("career_history", [])

    # 1. A role claims more tenure than has physically elapsed (+3mo slack)
    for r in ch:
        dur = float(r.get("duration_months", 0) or 0)
        elapsed = _elapsed_months(r.get("start_date", ""), r.get("end_date"))
        if dur > elapsed + 3:
            return True

    # 2. Many "expert" skills with zero months of actual use
    expert_zero = sum(1 for s in c.get("skills", [])
                      if (s.get("proficiency") == "expert"
                          and float(s.get("duration_months", 0) or 0) == 0))
    if expert_zero >= 3:
        return True

    # 3. Sum of stated tenures grossly exceeds total years of experience
    total_months = sum(float(r.get("duration_months", 0) or 0) for r in ch)
    yoe_months = float(c.get("profile", {}).get("years_of_experience", 0) or 0) * 12
    if yoe_months > 0 and total_months > yoe_months * 1.6 + 24:
        return True

    return False
