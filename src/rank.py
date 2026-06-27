"""Official ranking step: CPU-only, no network, <5 min. Loads candidates +
precomputed embeddings, scores everyone, ranks the top 100, writes a
validator-clean CSV. Single reproduce command lives in the README."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

from src import config as C
from src import io_utils, scoring, reasoning, precompute

HEADER = ["candidate_id", "rank", "score", "reasoning"]
SCORE_DECIMALS = 4

def select_top(scored, top_n: int = 100):
    """Round each score to the precision we actually write, then sort by that
    rounded score (desc) and candidate_id (asc). Sorting on the *rounded* value
    is what keeps the output consistent with the validator's tie-break rule:
    rows that print as equal scores must be ordered by candidate_id ascending.
    Adds a 'score4' field to each item and returns the top_n."""
    for s in scored:
        s["score4"] = round(float(s["score"]), SCORE_DECIMALS)
    scored.sort(key=lambda x: (-x["score4"], x["candidate_id"]))
    return scored[:top_n]

def rank(candidates_path: Path, artifacts_dir: Path, out_path: Path, top_n: int = 100):
    emb, ids, jd = precompute.load_artifacts(artifacts_dir)
    emb = emb.astype(np.float32)
    jd = jd.astype(np.float32)
    jd = jd / (np.linalg.norm(jd) + 1e-9)
    # embeddings already normalized at encode time, but renormalize defensively
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    sims = (emb / norms) @ jd                      # cosine per candidate

    id_to_row = {cid: i for i, cid in enumerate(ids)}

    scored = []
    for c in io_utils.iter_candidates(candidates_path):
        cid = c["candidate_id"]
        row = id_to_row.get(cid)
        if row is None:
            raise SystemExit(f"No precomputed embedding for {cid}; artifacts out of sync.")
        sem = float(sims[row])
        s = scoring.score_candidate(c, semantic=sem)
        s["_cand"] = c
        scored.append(s)

    top = select_top(scored, top_n)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, s in enumerate(top):
            rank_pos = i + 1
            reason = reasoning.build_reason(s["_cand"], s, rank=rank_pos, variant=i)
            # score non-increasing; equal rounded scores ordered by candidate_id asc.
            w.writerow([s["candidate_id"], rank_pos, f"{s['score4']:.4f}", reason])
    print(f"Wrote {len(top)} ranked candidates to {out_path}", file=sys.stderr)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(C.DATA_FILE))
    ap.add_argument("--artifacts", default=str(C.ARTIFACTS))
    ap.add_argument("--out", default=str(C.ROOT / "submission" / "submission.csv"))
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rank(Path(args.candidates), Path(args.artifacts), Path(args.out))

if __name__ == "__main__":
    main()
