"""Scoring invariants: the composite must respond monotonically to each signal
in the expected direction. These encode the methodology's contract."""
import copy
from src import scoring


def _base():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 7,
                    "location": "Pune, Maharashtra"},
        "career_history": [{"title": "ML Engineer", "company": "Flipkart",
                            "duration_months": 60, "is_current": True,
                            "start_date": "2021-06-01", "end_date": None,
                            "industry": "Software"}],
        "education": [{"tier": "tier_1"}],
        "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 40}],
        "redrob_signals": {"last_active_date": "2026-06-10", "recruiter_response_rate": 0.8,
                           "open_to_work_flag": True, "willing_to_relocate": True,
                           "interview_completion_rate": 0.9},
    }


def test_higher_semantic_never_lowers_score():
    c = _base()
    lo = scoring.score_candidate(c, semantic=0.3)["score"]
    hi = scoring.score_candidate(c, semantic=0.9)["score"]
    assert hi >= lo


def test_red_flag_lowers_score():
    clean = scoring.score_candidate(_base(), semantic=0.7)["score"]
    consulting = copy.deepcopy(_base())
    consulting["career_history"] = [
        {"title": "Consultant", "company": "Infosys", "duration_months": 60,
         "is_current": True, "start_date": "2021-06-01", "end_date": None,
         "industry": "IT Services"}]
    flagged = scoring.score_candidate(consulting, semantic=0.7)
    assert flagged["components"]["penalty"] > 0
    assert flagged["score"] < clean


def test_poor_availability_lowers_score():
    good = scoring.score_candidate(_base(), semantic=0.7)["score"]
    bad = copy.deepcopy(_base())
    bad["redrob_signals"].update({"last_active_date": "2025-01-01",
                                  "recruiter_response_rate": 0.02,
                                  "open_to_work_flag": False})
    assert scoring.score_candidate(bad, semantic=0.7)["score"] < good


def test_honeypot_forces_zero_even_with_perfect_signal():
    hp = copy.deepcopy(_base())
    hp["career_history"][0]["duration_months"] = 9999  # impossible tenure
    s = scoring.score_candidate(hp, semantic=0.99)
    assert s["is_honeypot"] is True
    assert s["score"] == 0.0
