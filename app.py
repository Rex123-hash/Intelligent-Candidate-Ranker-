"""Interactive sandbox for the Intelligent Candidate Ranker.

Runs the real ranking pipeline on a small candidate sample and shows the ranked
shortlist with scores and fact-grounded reasons. Uses the committed float16
embeddings, so the default demo needs no model and no network — exactly the
CPU-only, offline behaviour the challenge requires.

Run locally:   python app.py
Deploy:        works as the entry point of a HuggingFace Space (Gradio SDK).
"""
import json
from pathlib import Path

import numpy as np
import gradio as gr

from src import precompute, scoring, reasoning
from src.rank import select_top

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "demo" / "sample_candidates.jsonl"
ARTIFACTS = ROOT / "artifacts"

# --- load committed artifacts once ---
_emb, _ids, _jd = precompute.load_artifacts(ARTIFACTS)
_emb = _emb.astype(np.float32)
_jd = _jd.astype(np.float32)
_jd = _jd / (np.linalg.norm(_jd) + 1e-9)
_emb_norm = _emb / (np.linalg.norm(_emb, axis=1, keepdims=True) + 1e-9)
_sims = _emb_norm @ _jd
_ID_TO_ROW = {cid: i for i, cid in enumerate(_ids)}

HEADERS = ["rank", "candidate_id", "current title", "years", "score", "reasoning"]


def _load_sample():
    with open(SAMPLE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_candidates(candidates, top_n):
    """Score + rank an in-memory candidate list using precomputed similarities."""
    scored = []
    for c in candidates:
        row = _ID_TO_ROW.get(c["candidate_id"])
        if row is None:           # not in the precomputed pool — skip in this demo
            continue
        s = scoring.score_candidate(c, semantic=float(_sims[row]))
        s["_cand"] = c
        scored.append(s)
    top = select_top(scored, top_n=int(top_n))
    table = []
    for i, s in enumerate(top):
        p = s["_cand"].get("profile", {})
        reason = reasoning.build_reason(s["_cand"], s, rank=i + 1, variant=i)
        table.append([i + 1, s["candidate_id"], p.get("current_title", ""),
                      p.get("years_of_experience", ""), f"{s['score4']:.4f}", reason])
    return table


def run_sample(top_n):
    return rank_candidates(_load_sample(), top_n)


def run_upload(file, top_n):
    if file is None:
        return run_sample(top_n)
    with open(file.name, encoding="utf-8") as f:
        cands = [json.loads(line) for line in f if line.strip()]
    rows = rank_candidates(cands, top_n)
    return rows if rows else [["—", "no overlap with precomputed pool", "", "", "", ""]]


with gr.Blocks(title="Intelligent Candidate Ranker — Sandbox") as demo:
    gr.Markdown(
        "# Intelligent Candidate Ranker — Sandbox\n"
        "Ranks candidates for the **Senior AI Engineer** JD: semantic fit + structured "
        "JD features − red-flag penalties × behavioral availability, with honeypot demotion "
        "and fact-grounded reasoning. CPU-only, offline (uses committed embeddings)."
    )
    with gr.Row():
        top_n = gr.Slider(5, 50, value=10, step=1, label="Show top N")
    with gr.Tab("Sample pool (50 candidates)"):
        btn = gr.Button("Rank sample", variant="primary")
        out1 = gr.Dataframe(headers=HEADERS, wrap=True, label="Ranked shortlist")
        btn.click(run_sample, inputs=top_n, outputs=out1)
        demo.load(run_sample, inputs=top_n, outputs=out1)
    with gr.Tab("Upload your own (.jsonl)"):
        gr.Markdown("Upload a JSONL of candidates using `CAND_XXXXXXX` ids from the pool.")
        up = gr.File(file_types=[".jsonl"], label="candidates.jsonl")
        btn2 = gr.Button("Rank uploaded", variant="primary")
        out2 = gr.Dataframe(headers=HEADERS, wrap=True, label="Ranked shortlist")
        btn2.click(run_upload, inputs=[up, top_n], outputs=out2)


if __name__ == "__main__":
    demo.launch()
