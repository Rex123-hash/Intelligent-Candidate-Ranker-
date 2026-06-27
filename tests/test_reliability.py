"""Reliability / regression guards that protect the *actual* deliverable:
determinism, the integrity of the shipped submission file, reasoning diversity
(Stage-4 anti-templating), and config invariants."""
import csv
import re
import json
from pathlib import Path

import numpy as np
import pytest

from src import config as C
from src import scoring, reasoning, precompute

CID_RE = re.compile(r"^CAND_[0-9]{7}$")
SAMPLE = Path(__file__).parent / "fixtures_sample_candidates.json"
SUBMISSION = C.ROOT / "submission" / "submission.csv"


# ---------- config invariants ----------

def test_feature_weights_sum_to_one():
    total = (C.F_EXPERIENCE + C.F_PRODUCT + C.F_TITLE + C.F_LOCATION + C.F_EDUCATION)
    assert abs(total - 1.0) < 1e-9


def test_score_weights_cannot_exceed_unit_before_penalty():
    # semantic + features are both in [0,1]; their weighted sum must stay <= 1
    assert C.W_SEMANTIC + C.W_FEATURES <= 1.0 + 1e-9


# ---------- determinism ----------

def _candidate():
    return {"candidate_id": "CAND_0000001",
            "profile": {"current_title": "ML Engineer", "years_of_experience": 7,
                        "location": "Pune"},
            "career_history": [{"title": "ML Engineer", "company": "Flipkart",
                                "duration_months": 60, "is_current": True,
                                "start_date": "2021-06-01", "end_date": None}],
            "skills": [{"name": "NLP", "proficiency": "advanced", "duration_months": 40}],
            "education": [{"tier": "tier_1"}],
            "redrob_signals": {"last_active_date": "2026-06-10",
                               "recruiter_response_rate": 0.8, "open_to_work_flag": True}}


def test_scoring_is_deterministic():
    a = scoring.score_candidate(_candidate(), semantic=0.7)
    b = scoring.score_candidate(_candidate(), semantic=0.7)
    assert a == b


def test_reasoning_is_deterministic():
    c = _candidate()
    s = scoring.score_candidate(c, semantic=0.7)
    assert reasoning.build_reason(c, s, rank=1, variant=0) == \
           reasoning.build_reason(c, s, rank=1, variant=0)


# ---------- the actual shipped submission ----------

def test_shipped_submission_is_fully_valid():
    if not SUBMISSION.exists():
        pytest.skip("submission.csv not generated yet")
    rows = list(csv.DictReader(open(SUBMISSION, encoding="utf-8")))
    assert len(rows) == 100
    ids = [r["candidate_id"] for r in rows]
    ranks = [int(r["rank"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    assert all(CID_RE.match(i) for i in ids)            # id format
    assert len(set(ids)) == 100                          # unique ids
    assert sorted(ranks) == list(range(1, 101))          # ranks 1..100 once each
    assert all(scores[i] >= scores[i + 1] for i in range(99))  # non-increasing
    assert all(r["reasoning"].strip() for r in rows)     # every reason populated


# ---------- reasoning diversity (Stage 4 penalises templated reasons) ----------

def test_reasoning_is_diverse_on_real_sample():
    if not C.EMB_FILE.exists():
        pytest.skip("embedding artifacts not present")
    emb, ids, jd = precompute.load_artifacts(C.ARTIFACTS)
    emb = emb.astype(np.float32); jd = jd.astype(np.float32)
    jd = jd / (np.linalg.norm(jd) + 1e-9)
    sims = (emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)) @ jd
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    candidates = json.load(open(SAMPLE, encoding="utf-8"))

    reasons = []
    for n, c in enumerate(candidates):
        row = id_to_row.get(c["candidate_id"])
        if row is None:
            continue
        s = scoring.score_candidate(c, semantic=float(sims[row]))
        reasons.append(reasoning.build_reason(c, s, rank=n + 1, variant=n))
    # the reasons must not be near-identical boilerplate
    assert len(set(reasons)) >= max(3, int(0.6 * len(reasons)))
