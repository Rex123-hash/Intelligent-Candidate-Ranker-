# Intelligent Candidate Discovery & Ranking

> **Redrob × India Runs — Track 1 (Data & AI Challenge)**
> Rank the **top 100** best‑fit candidates from a **100,000‑profile** pool for a single, nuanced
> Senior AI Engineer job description — fast, offline, explainable, and resistant to keyword‑stuffing.

A transparent **hybrid ranker**: dense semantic understanding (so it reads *meaning*, not keywords)
combined with structured, JD‑derived judgment (experience, product‑vs‑services, title, location),
minus explicit red‑flag penalties, scaled by a behavioral availability signal, with impossible
"honeypot" profiles detected and demoted. No per‑candidate LLM calls. No labels were used (the ground
truth is hidden), so the scoring function is fully interpretable and defensible end‑to‑end.

---

## Results at a glance

| What | Result | Requirement |
|---|---|---|
| Ranking‑step runtime (full 100k) | **~12.6 s** on CPU | ≤ 5 min |
| Peak memory | well under 16 GB | ≤ 16 GB |
| Network during ranking | **none** | none allowed |
| GPU during ranking | **none** (pure NumPy) | none allowed |
| Format validation (`validate_submission.py`) | **passes** | hard gate |
| Honeypots in our top 100 | **0** (41 detected pool‑wide) | ≤ 10% |
| Automated tests | **29 passing** (incl. end‑to‑end validator check) | — |
| Embedding artifact | 73 MB (float16, committed) | reproducible offline |

---

## 1. The problem

Recruiters drown in profiles, and keyword/ATS filters miss real fits while surfacing keyword‑stuffers.
The task: read a complex JD, understand what it *means* (not just the words), evaluate every candidate
across profile, career metadata, and behavioral signals, and return a **lightning‑fast, accurately
ranked shortlist** of the top 100.

The released JD (Senior AI Engineer, Redrob) is deliberately nuanced. It explicitly states the right
answer is **not** "most AI keywords" — it even warns that the dataset contains traps. Our system is
built around that reality.

## 2. How submissions are scored (and how we optimise for it)

The hidden ground truth scores each submission with:

```
composite = 0.50 · NDCG@10  +  0.30 · NDCG@50  +  0.15 · MAP  +  0.05 · P@10
```

**Top‑10 quality dominates (50%)**, so our design prioritises getting the very top of the list right:
strong semantic match, correct seniority, genuine product‑company AI/ML experience, and high
availability — while pushing ambiguous or unavailable candidates down.

The evaluation then proceeds through code reproduction + honeypot check (Stage 3), manual review of
reasoning/methodology/code (Stage 4), and a defend‑your‑work interview (Stage 5). This repo is built to
clear all of them: reproducible within the compute budget, honest fact‑grounded reasoning, clean code,
and an architecture every component of which can be explained.

## 3. Compute constraints — and how we satisfy them

| Constraint | Limit | How we meet it |
|---|---|---|
| Runtime | ≤ 5 min | Ranking is pure NumPy over precomputed vectors → **~12.6 s** |
| Memory | ≤ 16 GB | Stream the 465 MB JSONL line‑by‑line; only a 73 MB matrix in RAM |
| Compute | CPU only | The ranking step loads **no model**; embeddings are precomputed offline |
| Network | off | No API calls at rank time — everything is local |
| Disk | ≤ 5 GB | Dataset 465 MB + artifacts 73 MB |

The expensive transformer work happens **once, offline** (`precompute.py`, GPU or CPU, no time limit).
The production‑shaped ranking step (`rank.py`) is a cheap, scalable vector pass — exactly the
latency/quality trade‑off the JD says the role cares about.

## 4. Approach & methodology

Two stages, split precisely because of the 5‑minute rule:

```
                 ┌──────────────────────── OFFLINE (one-time, GPU/CPU, unbounded) ────────────────────────┐
 candidates.jsonl ─▶ build profile text ─▶ bge-small-en-v1.5 ─▶ embeddings.npy (float16, 100k×384)
        JD.txt    ─▶ encode ───────────────────────────────────▶ jd_embedding.npy
                 └────────────────────────────────────────────────────────────────────────────────────────┘
                 ┌──────────────────── RANKING STEP (CPU only, no network, ~12.6 s) ─────────────────────┐
 candidates.jsonl + artifacts ─▶ cosine(JD, cand)  ┐
                                 structured features ┤
                                 red-flag penalties  ┼─▶ composite score ─▶ honeypot demotion
                                 behavioral modifier ┘        │
                                                              ▼
                                          sort (score↓, candidate_id↑) ─▶ top 100
                                                              ▼
                                          fact-grounded reasoning ─▶ submission.csv ─▶ submission.xlsx
                 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### The scoring function

For every candidate, five components combine into one interpretable score:

```
score = clip( W_sem·semantic  +  W_feat·features  −  W_pen·penalties , 0, 1 )  ×  behavior_modifier
        (honeypots forced to 0)
```

with `W_sem = 0.45`, `W_feat = 0.45`, `W_pen = 0.30` (all in `src/config.py`).

1. **Semantic fit** — cosine similarity between the candidate's profile embedding and the JD embedding.
   Captures transferable experience with no shared keywords (e.g. *"built a recommendation engine"* ≈
   retrieval/ranking experience).
2. **Structured JD‑fit features** (weighted mean, each in `[0,1]`):
   experience band (peak at 6–8 yrs · `0.25`), product‑company vs pure‑services history (`0.25`),
   title relevance (`0.30`), location — Pune/Noida/Hyderabad/Mumbai/Delhi‑NCR or willing to relocate
   (`0.12`), education tier (`0.08`).
3. **Red‑flag penalties** (subtractive) — the JD's explicit *anti‑requirements*: career entirely at
   consulting/services firms; AI experience that is only recent LLM‑wrapper work; senior management
   title with no recent hands‑on engineering; vision/speech/robotics without NLP/IR; title‑chasing
   (frequent short stints).
4. **Behavioral availability modifier** (multiplicative, floor `0.55`) — down‑weights candidates who
   aren't realistically hireable: stale last‑active date, low recruiter response rate, not open to work.
5. **Honeypot demotion** — see §6.

### Tie‑breaking (a subtle but important detail)

Scores are written at 4 decimals, so the tie‑break sorts on the **rounded** value, then `candidate_id`
ascending — exactly matching the validator's rule. (Sorting on full precision would let two rows print
as equal yet be ordered by score, which the validator rejects.) See `src/rank.py:select_top`.

## 5. Reading the JD's *meaning*, not its keywords

The JD is explicit that keyword‑matching is a trap. Our features and penalties encode its intent:

| The JD wants | We reward |
|---|---|
| 6–8 yrs applied ML at **product** companies | experience band + product‑vs‑services share |
| Embeddings/retrieval/ranking experience | semantic similarity + title relevance |
| Reachable, in‑market candidates | behavioral availability modifier |
| Pune/Noida (or relocation) | location feature |

| The JD rejects | We penalise / demote |
|---|---|
| Pure consulting‑firm careers | consulting‑only penalty |
| <12‑mo LangChain‑only "AI experience" | LLM‑wrapper‑only penalty |
| Managers who stopped coding | management‑no‑recent‑code penalty |
| Vision/speech‑only backgrounds | CV/speech‑without‑NLP penalty |
| Title‑chasers | short‑stint penalty |
| **Keyword‑stuffers** (e.g. "Marketing Manager" with AI skills) | low title relevance + honeypot checks |

## 6. Honeypots & data validation

The dataset seeds ~80 subtly **impossible** profiles (forced to relevance tier 0); ranking >10% of them
in the top 100 is an automatic disqualification. `src/honeypot.py` flags them with conservative,
false‑positive‑averse rules:

- a role claiming more tenure than has physically elapsed since its start date;
- many "expert" skills with 0 months of actual use;
- total stated tenure grossly inconsistent with `years_of_experience`.

Detected impossibilities are demoted below the cutoff. **Result: 0 honeypots in our top 100** (41 caught
across the full pool). We deliberately don't over‑special‑case — a sound ranker should *naturally* avoid
them, and ours does.

## 7. Explainability (no hallucinations)

Every candidate gets a 1–2 sentence reason **assembled from facts actually present in their profile** —
years, current title, named skills, product background, and the single biggest concern — never free
text. A unit test asserts no out‑of‑profile skill can appear in a reason. Tone tracks rank, and phrasing
varies across candidates to avoid templated output. Example:

> *"6.4 yrs as ML Engineer; core skills NLP, PyTorch; product‑company background; strong semantic match. Concern: long notice period (120 days)."*

## 8. Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.13
pip install -r requirements.txt

# (offline, one-time) regenerate embeddings — OPTIONAL; artifacts are committed:
python -m src.precompute --candidates ./data/candidates.jsonl --jd ./data/jd.txt --out ./artifacts

# ranking step — CPU only, no network, < 5 min:
python -m src.rank --candidates ./data/candidates.jsonl --artifacts ./artifacts --out ./submission/submission.csv

# validate the format, then export the XLSX for the portal:
python validate_submission.py ./submission/submission.csv
python -m src.export_xlsx --csv ./submission/submission.csv --out ./submission/submission.xlsx
```

**Single reproduce command (for Stage 3):**

```bash
python -m src.rank --candidates ./data/candidates.jsonl --artifacts ./artifacts --out ./submission/submission.csv
```

The committed `artifacts/` let this run with no GPU and no network, inside the evaluation sandbox.

## 9. Repository layout

```
src/config.py        all tunable weights + JD-derived lexicons (single source of truth)
src/io_utils.py      streaming JSONL loader + profile-text builder
src/features.py      structured JD-fit feature scores
src/flags.py         red-flag penalties (JD anti-requirements)
src/signals.py       behavioral availability modifier
src/honeypot.py      impossible-profile detection
src/scoring.py       composite score = (semantic + features − penalties) × behavior
src/reasoning.py     fact-grounded per-candidate reasoning
src/precompute.py    offline embedding precompute → artifacts/
src/rank.py          CPU-only ranking driver → validator-clean top-100 CSV
src/export_xlsx.py   CSV → XLSX for portal upload
artifacts/           committed float16 embeddings + JD embedding + candidate ids
data/jd.txt          the job description used for ranking
tests/               29 tests, including an end-to-end validator check
submission/          submission.csv (canonical) + submission.xlsx (portal)
validate_submission.py   organizers' validator (used in tests and locally)
submission_metadata.yaml record of team, compute, and declarations
```

## 10. Testing

```bash
pytest        # 29 tests
```

Coverage includes feature logic, red‑flag detection, behavioral scoring, honeypot detection, the
rounding tie‑break, reasoning anti‑hallucination, and an **end‑to‑end test that runs `rank.py` and
confirms the output passes the organizers' `validate_submission.py`.**

## 11. Design decisions & trade‑offs

- **No supervised learning‑to‑rank.** The ground truth is hidden — there are no labels to train on.
  A trained ranker would be guesswork *and* hard to defend. A transparent scoring function is honest,
  interpretable, and directly justifiable at the Stage 5 interview.
- **No per‑candidate LLM calls.** Banned during ranking and unable to fit the compute budget at 100k
  scale. Embeddings are precomputed once; ranking is vector math.
- **`bge-small-en-v1.5`.** Small, fast, strong retrieval quality; runs locally with no API; 384‑dim
  vectors store compactly as float16 (73 MB for 100k, under GitHub's 100 MB limit).
- **All weights in one config file.** Every number is named, readable, and tunable — and explainable.

## 12. Reproducibility & compute declaration

- Embeddings are precomputed offline (one‑time, ~17 min on an RTX 4060) and **committed**, so the
  ranking step needs no GPU and no network.
- The ranking step is deterministic given the same candidates file and artifacts.
- Tested end‑to‑end on a 16 GB, CPU‑only configuration within the 5‑minute budget.

## 13. Limitations & future work

- Honeypot detection is intentionally conservative (catches 41 of ~80) to avoid false positives that
  would hurt NDCG; the broader scorer keeps the rest out of the top ranks regardless.
- Feature weights are hand‑justified from the JD rather than tuned against labels (none exist); a small
  held‑out human relevance set would allow principled weight calibration.
- A hosted demo (HuggingFace Space / Streamlit) can wrap `rank.py` for an interactive small‑sample run.

## 14. AI tooling (transparency)

An AI coding assistant was used for parts of implementation and code review. **No candidate data was
sent to any hosted LLM**, and the ranking step makes **no network calls** — it scores candidates locally
over precomputed embeddings. All design decisions are documented here and defensible.

---

*Team: WebHackers · Track 1 — Intelligent Candidate Discovery & Ranking.*
