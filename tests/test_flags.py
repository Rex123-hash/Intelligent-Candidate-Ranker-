# tests/test_flags.py
from src import flags

def test_consulting_only_penalised():
    ch = [{"company": "Infosys", "duration_months": 36, "industry": "IT Services"},
          {"company": "Wipro", "duration_months": 36, "industry": "IT Services"}]
    pen, reasons = flags.penalties({"profile": {"current_title": "Software Engineer",
        "years_of_experience": 6}, "career_history": ch, "skills": []})
    assert pen > 0
    assert any("consult" in r.lower() or "services" in r.lower() for r in reasons)

def test_manager_no_recent_code_penalised():
    ch = [{"title": "Engineering Manager", "duration_months": 24, "is_current": True},
          {"title": "Director of Engineering", "duration_months": 24, "is_current": False}]
    pen, reasons = flags.penalties({"profile": {"current_title": "Engineering Manager",
        "years_of_experience": 12}, "career_history": ch, "skills": []})
    assert pen > 0

def test_clean_profile_no_penalty():
    ch = [{"title": "ML Engineer", "company": "Flipkart", "duration_months": 48,
           "is_current": True, "industry": "Software"}]
    pen, reasons = flags.penalties({"profile": {"current_title": "ML Engineer",
        "years_of_experience": 7}, "career_history": ch,
        "skills": [{"name": "PyTorch", "duration_months": 40}]})
    assert pen == 0
    assert reasons == []
