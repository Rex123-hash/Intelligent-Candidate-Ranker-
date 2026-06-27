"""Guard: a duplicate candidate_id in the input pool must never produce a
duplicate in the ranked output (which the validator would reject)."""
import csv
import json
import subprocess
import sys

import numpy as np

from src import precompute, rank


def _cand(i):
    cid = f"CAND_{i:07d}"
    return {"candidate_id": cid,
            "profile": {"current_title": "ML Engineer", "years_of_experience": 7,
                        "location": "Pune", "current_company": "Flipkart",
                        "current_industry": "Software", "headline": "ML Engineer",
                        "summary": "x"},
            "career_history": [{"title": "ML Engineer", "company": "Flipkart",
                                "duration_months": 84, "is_current": True,
                                "start_date": "2020-01-01", "end_date": None,
                                "industry": "Software", "company_size": "1001-5000",
                                "description": "built ranking systems"}],
            "education": [{"tier": "tier_1"}],
            "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 40}],
            "redrob_signals": {"last_active_date": "2026-06-10", "open_to_work_flag": True,
                               "recruiter_response_rate": 0.8, "willing_to_relocate": True,
                               "interview_completion_rate": 0.9}}


def test_duplicate_input_ids_produce_unique_output(tmp_path):
    # CAND_0000001 appears 3 times; plenty of others to fill 100
    cands = [_cand(1), _cand(1), _cand(1)] + [_cand(i) for i in range(2, 140)]
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text("\n".join(json.dumps(c) for c in cands), encoding="utf-8")

    # unique-keyed artifacts
    uids, rows = [], []
    rng = np.random.default_rng(0)
    for c in cands:
        if c["candidate_id"] not in uids:
            uids.append(c["candidate_id"])
            rows.append(rng.random(8))
    precompute.save_artifacts(tmp_path, np.array(rows, dtype=np.float32), uids,
                              np.ones(8, dtype=np.float32))

    out = tmp_path / "team.csv"
    rank.rank(jsonl, tmp_path, out, top_n=100)

    ids = [r["candidate_id"] for r in csv.DictReader(open(out, encoding="utf-8"))]
    assert len(ids) == 100
    assert len(set(ids)) == 100, "duplicate candidate_id leaked into output"

    # and the organizers' validator accepts it
    v = subprocess.run([sys.executable, "validate_submission.py", str(out)],
                       capture_output=True, text=True)
    assert "Submission is valid." in v.stdout, v.stdout + v.stderr
