# tests/test_reasoning.py
from src import reasoning

def _cand_and_score():
    c = {"candidate_id": "CAND_0000001",
         "profile": {"current_title": "ML Engineer", "years_of_experience": 6.4,
                     "location": "Pune, Maharashtra", "current_company": "Flipkart"},
         "career_history": [{"title": "ML Engineer", "company": "Flipkart",
                             "duration_months": 60, "is_current": True}],
         "skills": [{"name": "NLP"}, {"name": "PyTorch"}],
         "redrob_signals": {"notice_period_days": 120, "recruiter_response_rate": 0.8,
                            "last_active_date": "2026-06-10"}}
    sc = {"components": {"semantic": 0.81, "title": 0.9, "product": 1.0,
                         "experience": 0.9, "location": 1.0, "penalty": 0.0,
                         "behavior": 0.95}, "flag_reasons": []}
    return c, sc

def test_reasoning_mentions_real_facts_only():
    c, sc = _cand_and_score()
    text = reasoning.build_reason(c, sc, rank=3, variant=0)
    assert "6.4" in text                      # real years
    assert "ML Engineer" in text or "ml engineer" in text.lower()
    # must not invent a skill the candidate lacks
    assert "kubernetes" not in text.lower()
    assert 0 < len(text) <= 240

def test_reasoning_surfaces_concern_when_present():
    c, sc = _cand_and_score()                 # notice_period 120 days
    text = reasoning.build_reason(c, sc, rank=3, variant=0)
    assert "notice" in text.lower()

def test_variants_differ():
    c, sc = _cand_and_score()
    a = reasoning.build_reason(c, sc, rank=3, variant=0)
    b = reasoning.build_reason(c, sc, rank=4, variant=1)
    assert a != b
