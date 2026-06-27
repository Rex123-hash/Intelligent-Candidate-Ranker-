# Intelligent Candidate Ranker — Redrob India Runs (Track 1)

Ranks the top 100 candidates from `candidates.jsonl` (100,000 profiles) for the
released **Senior AI Engineer** job description, with a 1–2 sentence, fact-grounded
reason for each pick.

## Approach (one paragraph)

Two stages. **Offline** (`precompute.py`, GPU or CPU, no time limit) we embed every
candidate's profile text and the JD with `BAAI/bge-small-en-v1.5` and store compact
float16 vectors as artifacts. The **ranking step** (`rank.py`) is CPU-only and offline:
for each candidate it combines (1) semantic similarity to the JD, (2) structured JD-fit
features — experience band, product-company vs pure-services history, title relevance,
location, education — minus (3) red-flag penalties (consulting-only career,
LLM-wrapper-only AI experience, management title with no recent hands-on engineering,
vision/speech-only background, title-chasing), all multiplied by (4) a behavioral
availability modifier (recency, recruiter response rate, open-to-work). Subtly
**impossible "honeypot" profiles are detected and demoted** below the cutoff. Reasoning
is generated from facts actually present in each profile — no hallucinated skills.
There are no labels in the data, so this is a transparent, defensible scoring function,
not a trained model.

## Compute (matches the challenge constraints)

The ranking step is **CPU only, makes no network calls, uses < 16 GB RAM, and completes
in well under 5 minutes**. Embeddings are precomputed offline and committed under
`artifacts/`, so the ranking step loads them with no model and no internet.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate     # Python 3.13
pip install -r requirements.txt

# (offline, one-time) regenerate embeddings — OPTIONAL, artifacts are committed:
python -m src.precompute --candidates ./data/candidates.jsonl --jd ./data/jd.txt --out ./artifacts

# ranking step (CPU only, < 5 min, no network):
python -m src.rank --candidates ./data/candidates.jsonl --artifacts ./artifacts --out ./submission/submission.csv

# validate format + export the XLSX for the portal:
python validate_submission.py ./submission/submission.csv
python -m src.export_xlsx --csv ./submission/submission.csv --out ./submission/submission.xlsx
```

## Layout

```
src/config.py       weights + JD-derived lexicons (all tunables live here)
src/io_utils.py     streaming JSONL loader + profile-text builder
src/features.py     structured JD-fit feature scores
src/flags.py        red-flag penalties (JD anti-requirements)
src/signals.py      behavioral availability modifier
src/honeypot.py     impossible-profile detection
src/scoring.py      composite score (semantic + features - penalties) x behavior
src/reasoning.py    fact-grounded per-candidate reasoning
src/precompute.py   offline embedding precompute -> artifacts/
src/rank.py         CPU-only ranking driver -> top-100 CSV
src/export_xlsx.py  CSV -> XLSX for portal upload
artifacts/          committed float16 embeddings + JD embedding + candidate ids
```

## Tests

```bash
pytest        # 27 tests; includes an end-to-end check that the output CSV passes
              # the organizers' validate_submission.py
```
