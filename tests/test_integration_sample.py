"""Real-data integration test: run the full scoring pipeline over the 50-candidate
sample using the committed embedding artifacts, and assert the system's core
guarantees hold on genuine data (score bounds, no-hallucination reasoning, and a
validator-consistent ordering). Skips gracefully if artifacts aren't present."""
import json
from pathlib import Path

import numpy as np
import pytest

from src import config as C
from src import precompute, scoring, reasoning
from src.rank import select_top

SAMPLE = Path(__file__).parent / "fixtures_sample_candidates.json"


def _load():
    if not C.EMB_FILE.exists():
        pytest.skip("embedding artifacts not present; run precompute first")
    emb, ids, jd = precompute.load_artifacts(C.ARTIFACTS)
    emb = emb.astype(np.float32)
    jd = jd.astype(np.float32)
    jd = jd / (np.linalg.norm(jd) + 1e-9)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    sims = (emb / norms) @ jd
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    candidates = json.load(open(SAMPLE, encoding="utf-8"))
    return sims, id_to_row, candidates


def test_pipeline_runs_on_real_sample_with_bounded_scores():
    sims, id_to_row, candidates = _load()
    scored = []
    for c in candidates:
        row = id_to_row.get(c["candidate_id"])
        if row is None:
            continue
        s = scoring.score_candidate(c, semantic=float(sims[row]))
        assert 0.0 <= s["score"] <= 1.0
        s["_cand"] = c
        scored.append(s)
    assert len(scored) > 0


def test_reasoning_never_hallucinates_skills_on_real_sample():
    sims, id_to_row, candidates = _load()
    for c in candidates:
        row = id_to_row.get(c["candidate_id"])
        if row is None:
            continue
        s = scoring.score_candidate(c, semantic=float(sims[row]))
        text = reasoning.build_reason(c, s, rank=1, variant=0)
        # any skill named in the "core skills ..." clause must exist in the profile
        profile_skills = {sk.get("name", "").lower() for sk in c.get("skills", [])}
        if "core skills " in text:
            listed = text.split("core skills ", 1)[1].split(";")[0]
            for name in [x.strip().rstrip(".").lower() for x in listed.split(",")]:
                assert name in profile_skills, f"hallucinated skill '{name}' for {c['candidate_id']}"


def test_select_top_ordering_is_validator_consistent_on_real_sample():
    sims, id_to_row, candidates = _load()
    scored = []
    for c in candidates:
        row = id_to_row.get(c["candidate_id"])
        if row is None:
            continue
        s = scoring.score_candidate(c, semantic=float(sims[row]))
        scored.append(s)
    top = select_top(scored, top_n=len(scored))
    # score4 strictly non-increasing; equal score4 ordered by candidate_id ascending
    for a, b in zip(top, top[1:]):
        assert a["score4"] >= b["score4"]
        if a["score4"] == b["score4"]:
            assert a["candidate_id"] <= b["candidate_id"]
