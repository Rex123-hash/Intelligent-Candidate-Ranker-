# tests/test_honeypot.py
from src import honeypot

def test_impossible_current_tenure_flagged():
    # current role started 2024 but claims 96 months (8 yrs) tenure -> impossible by 2026-06
    c = {"profile": {"years_of_experience": 8},
         "career_history": [{"start_date": "2024-01-01", "end_date": None,
                             "is_current": True, "duration_months": 96}],
         "skills": []}
    assert honeypot.is_honeypot(c) is True

def test_many_expert_zero_months_flagged():
    c = {"profile": {"years_of_experience": 6}, "career_history": [
            {"start_date": "2020-01-01", "end_date": None, "is_current": True,
             "duration_months": 70}],
         "skills": [{"name": s, "proficiency": "expert", "duration_months": 0}
                    for s in ("A", "B", "C", "D")]}
    assert honeypot.is_honeypot(c) is True

def test_normal_profile_not_flagged():
    c = {"profile": {"years_of_experience": 6.9},
         "career_history": [{"start_date": "2024-03-08", "end_date": None,
                             "is_current": True, "duration_months": 27},
                            {"start_date": "2019-07-03", "end_date": "2024-01-08",
                             "is_current": False, "duration_months": 55}],
         "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 26}]}
    assert honeypot.is_honeypot(c) is False
