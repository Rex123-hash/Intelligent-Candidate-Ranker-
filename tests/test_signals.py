# tests/test_signals.py
from src import signals
from src import config as C

def _sig(**kw):
    base = {"last_active_date": "2026-06-01", "recruiter_response_rate": 0.8,
            "open_to_work_flag": True, "interview_completion_rate": 0.9}
    base.update(kw)
    return {"redrob_signals": base}

def test_active_responsive_candidate_near_one():
    assert signals.behavior_modifier(_sig()) > 0.9

def test_inactive_unresponsive_candidate_downweighted():
    m = signals.behavior_modifier(_sig(last_active_date="2025-10-01",
                                       recruiter_response_rate=0.05,
                                       open_to_work_flag=False))
    assert C.BEHAVIOR_FLOOR <= m < 0.75

def test_modifier_always_in_range():
    m = signals.behavior_modifier(_sig(last_active_date="2024-01-01",
                                       recruiter_response_rate=0.0))
    assert C.BEHAVIOR_FLOOR <= m <= 1.0
