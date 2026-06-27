# tests/test_rank_e2e.py
import json, subprocess, sys
import numpy as np
from pathlib import Path
from src import precompute

def _make_candidate(i, title, yoe):
    cid = f"CAND_{i:07d}"
    return {"candidate_id": cid,
            "profile": {"anonymized_name": f"N{i}", "current_title": title,
                        "years_of_experience": yoe, "location": "Pune, Maharashtra",
                        "current_company": "Flipkart", "current_industry": "Software",
                        "headline": title, "summary": f"{title} summary"},
            "career_history": [{"title": title, "company": "Flipkart",
                                "duration_months": int(yoe*12), "is_current": True,
                                "start_date": "2020-01-01", "end_date": None,
                                "industry": "Software", "company_size": "1001-5000",
                                "description": "built ranking systems"}],
            "education": [{"institution": "IIT", "degree": "BE", "field_of_study": "CS",
                           "start_year": 2010, "end_year": 2014, "tier": "tier_1"}],
            "skills": [{"name": "NLP", "proficiency": "advanced", "endorsements": 5,
                        "duration_months": 40}],
            "redrob_signals": {"profile_completeness_score": 90, "signup_date": "2024-01-01",
                               "last_active_date": "2026-06-10", "open_to_work_flag": True,
                               "profile_views_received_30d": 5, "applications_submitted_30d": 2,
                               "recruiter_response_rate": 0.8, "avg_response_time_hours": 3,
                               "skill_assessment_scores": {}, "connection_count": 100,
                               "endorsements_received": 20, "notice_period_days": 30,
                               "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
                               "preferred_work_mode": "hybrid", "willing_to_relocate": True,
                               "github_activity_score": 50, "search_appearance_30d": 3,
                               "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.9,
                               "offer_acceptance_rate": 0.5, "verified_email": True,
                               "verified_phone": True, "linkedin_connected": True}}

def test_rank_produces_valid_csv(tmp_path):
    # 120 candidates so we can take a top-100
    cands = [_make_candidate(i, "ML Engineer" if i % 2 else "Marketing Manager", 7)
             for i in range(1, 121)]
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text("\n".join(json.dumps(c) for c in cands), encoding="utf-8")

    # fake aligned artifacts: dim 8; ML titles get a JD-aligned vector
    rng = np.random.default_rng(0)
    jd = np.zeros(8, dtype=np.float32); jd[0] = 1.0
    emb = np.zeros((120, 8), dtype=np.float32)
    for k, c in enumerate(cands):
        v = rng.random(8) * 0.1
        if "ML" in c["profile"]["current_title"]:
            v[0] = 0.9
        emb[k] = v / (np.linalg.norm(v) + 1e-9)
    precompute.save_artifacts(tmp_path, emb, [c["candidate_id"] for c in cands], jd)

    out = tmp_path / "team_test.csv"
    rc = subprocess.run([sys.executable, "-m", "src.rank", "--candidates", str(jsonl),
                         "--artifacts", str(tmp_path), "--out", str(out)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr

    # validate with the organizers' validator
    v = subprocess.run([sys.executable, "validate_submission.py", str(out)],
                       capture_output=True, text=True)
    assert "Submission is valid." in v.stdout, v.stdout + v.stderr
