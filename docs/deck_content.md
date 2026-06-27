# Deck Content — paste into `Idea Submission Template _ Redrob.pptx`

Use the organizers' template as-is (11 slides). Paste each section below into the
matching slide. `[[FILL ...]]` markers are values to drop in after the final run.

---

## Slide 1 — Title
- **Team Name:** WebHackers
- **Problem Statement:** Data & AI Challenge — Intelligent Candidate Discovery & Ranking
- **Team Leader Name:** [[FILL: leader name]]

---

## Slide 2 — Solution Overview
**What is your proposed solution?**
- A CPU-only batch ranker that reads the 100,000-candidate pool and returns the top 100
  best-fit candidates for the Senior AI Engineer JD, each with an honest 1–2 sentence reason.
- Two stages: embeddings are precomputed offline; the ranking step is pure, fast, offline math.

**What differentiates it from traditional candidate matching?**
- Goes beyond keyword/ATS matching: semantic understanding of meaning (e.g. "built a
  recommendation engine" ≈ retrieval/ranking experience, even without the buzzwords).
- Combines *meaning* with *structured judgment*: it actively penalizes the JD's stated
  anti-patterns (consulting-only careers, LLM-wrapper-only experience, managers who stopped
  coding) and down-weights candidates who aren't realistically available.
- Detects and demotes the dataset's hidden impossible "honeypot" profiles instead of being
  fooled by keyword-stuffing.

---

## Slide 3 — JD Understanding & Candidate Evaluation
**Key requirements extracted from the JD:**
- 6–8 years ideal (5–9 band); applied ML at **product** companies, not pure services.
- Production embeddings/retrieval + vector-search experience; ranking-evaluation literacy
  (NDCG/MRR/MAP); strong Python; shipper mindset over pure research.
- Explicit anti-requirements: pure research, <12mo LangChain-only, no code in 18 months,
  title-chasing, consulting-only careers (TCS/Infosys/Wipro/…), CV/speech-only.
- Location: Pune/Noida/Hyderabad/Mumbai/Delhi-NCR or willing to relocate.

**Which candidate signals matter most:**
- Title + career trajectory (product vs services), real experience band, semantic JD match.
- Behavioral availability: last-active recency, recruiter response rate, open-to-work.

**Fit beyond keyword matching:**
- Semantic similarity captures transferable experience; structured features and red-flag
  penalties encode the JD's *meaning*, not its vocabulary.

---

## Slide 4 — Ranking Methodology
**Retrieve → score → rank:**
- Offline: encode each candidate's profile text + the JD with `BAAI/bge-small-en-v1.5`.
- Online (CPU): cosine similarity gives the semantic signal; combine with structured features.

**Models / algorithms / heuristics:**
- Sentence-transformer embeddings (semantic) + transparent, hand-justified feature scores.
- No supervised model — the ground truth is hidden (no labels), so a defensible scoring
  function is the honest choice (and is fully explainable).

**How signals combine into a final score:**
- `score = clip(0.45·semantic + 0.45·features − 0.30·penalties, 0, 1) × behavior_modifier`
- Features = weighted mean of experience(0.25), product-vs-services(0.25), title(0.30),
  location(0.12), education(0.08).
- Honeypots forced to ~0. Ties broken by candidate_id ascending (matches the validator).

---

## Slide 5 — Explainability & Data Validation
**How ranking decisions are explained:**
- Every candidate's reason is assembled from the exact facts that drove its score
  (years, current title, named skills, product background, the single biggest concern).

**How we prevent hallucinations / unsupported justifications:**
- Reasoning only references fields present in that candidate's profile — no invented skills
  or employers. A unit test asserts no out-of-profile skill can appear.

**How we handle inconsistent / low-quality / suspicious profiles:**
- Honeypot detection flags impossible profiles (tenure exceeding elapsed time; many "expert"
  skills with 0 months used; career durations inconsistent with stated experience) and demotes
  them below the top-100 cutoff — directly targeting the dataset's keyword-stuffer traps.

---

## Slide 6 — End-to-End Workflow
1. **Input:** `candidates.jsonl` (100k profiles) + the fixed JD.
2. **Offline prep:** build profile text per candidate → embed candidates + JD → save float16
   artifacts (one-time, no time limit).
3. **Ranking (CPU, offline, <5 min):** load artifacts → semantic similarity → structured
   features − penalties → × behavioral modifier → honeypot demotion.
4. **Rank & select:** sort by score (tie-break candidate_id asc) → take top 100.
5. **Explain:** generate a fact-grounded reason per candidate.
6. **Output:** validator-clean `submission.csv` → exported to `submission.xlsx` for the portal.

---

## Slide 7 — System Architecture
- **Stage A — Offline precompute (`precompute.py`):** `candidates.jsonl` → profile text →
  `bge-small-en-v1.5` encoder → `embeddings.npy` (float16, ~77 MB) + `jd_embedding.npy` +
  `candidate_ids.npy`. Runs once; can use GPU; no time limit.
- **Stage B — Ranking (`rank.py`, CPU only, no network, <5 min):** loads artifacts → NumPy
  cosine similarity → `scoring.py` (features + flags + signals + honeypot) → sort/tie-break →
  `reasoning.py` → writes top-100 CSV. `export_xlsx.py` mirrors CSV → XLSX.
- Design rationale: the heavy model work is offline, so the in-production ranking step is a
  cheap, scalable NumPy pass — exactly the latency/quality tradeoff the JD cares about.
  *(Draw this as two boxes A→B sharing the artifacts store.)*

---

## Slide 8 — Results & Performance
- **Ranking-step runtime:** [[FILL: measured seconds]] for the full 100k pool on CPU —
  comfortably within the 5-minute budget (≤16 GB RAM, no network, no GPU).
- **Format:** output passes the organizers' `validate_submission.py` (100 rows, unique ranks,
  non-increasing scores, correct tie-break).
- **Honeypots in top 100:** [[FILL: count]] (target ≤10% — well clear of the disqualification line).
- **Top-of-list quality:** top ranks are product-company AI/ML profiles with strong semantic
  match and high availability; off-target titles (e.g. Marketing Manager) are pushed down.
- 27 automated tests pass, including the end-to-end format check.

---

## Slide 9 — Technologies Used
- **Python 3.13** — language; standard `csv`/`json` for I/O.
- **sentence-transformers + PyTorch** (`bge-small-en-v1.5`) — offline semantic embeddings;
  small, fast, strong retrieval quality; runs locally with no API.
- **NumPy** — the entire ranking step (cosine + scoring) as fast vectorized math, so it fits
  the CPU/5-minute budget.
- **openpyxl** — CSV → XLSX export for the portal.
- **pytest** — TDD; 27 tests including an end-to-end validator check.
- **Git** — full, incremental commit history.

---

## Slide 10 — Submission Assets
- **GitHub repo (public):** [[FILL: repo URL]] — code, tests, README, committed artifacts.
- **Ranked output:** `submission.xlsx` (top 100, with reasoning).
- **This deck (PDF).**
- *(Optional)* demo/sandbox link: [[FILL if built]].

---

## Slide 11 — Closing
- WebHackers — Intelligent Candidate Discovery & Ranking.
- "Match meaning, not keywords — and only surface candidates you can actually hire."
