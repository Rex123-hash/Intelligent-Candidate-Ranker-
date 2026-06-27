"""Combine semantic similarity + structured features - penalties, times the
behavioral modifier, with honeypot demotion. Returns score + component breakdown
(the breakdown feeds reasoning generation)."""
from typing import Dict, Any
from src import config as C
from src import features as F
from src import flags as FL
from src import signals as SG
from src import honeypot as HP

def score_candidate(c: Dict[str, Any], semantic: float) -> Dict[str, Any]:
    """semantic: cosine similarity in [~ -1, 1], already computed upstream."""
    sem = max(0.0, min(1.0, (semantic + 1.0) / 2.0)) if semantic < 0 else min(1.0, semantic)
    feats = F.feature_subscore(c)
    penalty, flag_reasons = FL.penalties(c)
    behavior = SG.behavior_modifier(c)
    honey = HP.is_honeypot(c)

    raw = C.W_SEMANTIC * sem + C.W_FEATURES * feats["aggregate"] - C.W_PENALTY * penalty
    raw = max(0.0, min(1.0, raw))
    score = raw * behavior
    if honey:
        score = 0.0

    return {
        "candidate_id": c["candidate_id"],
        "score": round(score, 6),
        "is_honeypot": honey,
        "components": {
            "semantic": round(sem, 4),
            **{k: round(v, 4) for k, v in feats.items()},
            "penalty": round(penalty, 4),
            "behavior": round(behavior, 4),
        },
        "flag_reasons": flag_reasons,
    }
