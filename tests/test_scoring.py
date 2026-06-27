# tests/test_scoring.py
from src import scoring

def _candidate(title="ML Engineer", yoe=7, company="Flipkart", relocate=True):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"current_title": title, "years_of_experience": yoe,
                    "location": "Pune, Maharashtra"},
        "career_history": [{"title": title, "company": company, "duration_months": 60,
                            "is_current": True, "start_date": "2021-06-01",
                            "end_date": None, "industry": "Software"}],
        "education": [{"tier": "tier_1"}],
        "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 40},
                   {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36}],
        "redrob_signals": {"last_active_date": "2026-06-10", "recruiter_response_rate": 0.8,
                           "open_to_work_flag": True, "willing_to_relocate": relocate,
                           "interview_completion_rate": 0.9},
    }

def test_score_in_unit_range():
    s = scoring.score_candidate(_candidate(), semantic=0.7)
    assert 0.0 <= s["score"] <= 1.0
    assert "components" in s

def test_strong_beats_weak():
    strong = scoring.score_candidate(_candidate("Machine Learning Engineer"), semantic=0.8)["score"]
    weak = scoring.score_candidate(_candidate("Marketing Manager", company="Infosys"),
                                   semantic=0.2)["score"]
    assert strong > weak

def test_honeypot_forced_low():
    hp = _candidate()
    hp["career_history"][0]["duration_months"] = 999   # impossible tenure
    s = scoring.score_candidate(hp, semantic=0.9)
    assert s["score"] <= 0.01
    assert s["is_honeypot"] is True
