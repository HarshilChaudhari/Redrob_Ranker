# INDIA RUNS Track — Redrob AI Candidate Ranking

**Team:** your-team-name-here

A deterministic, rule-based candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge. Scores ~100K candidates for a **Senior AI Engineer — Founding Team** role using 6 weighted components derived from skills, career history, title, location, and behavioral signals. No ML models, no embeddings, no LLMs — runs in ~5 seconds on CPU.

---

## Table of Contents

1. [Solution Overview](#1-solution-overview)
2. [JD Understanding & Candidate Evaluation](#2-jd-understanding--candidate-evaluation)
3. [Ranking Methodology](#3-ranking-methodology)
4. [Explainability & Data Validation](#4-explainability--data-validation)
5. [End-to-End Workflow](#5-end-to-end-workflow)
6. [System Architecture](#6-system-architecture)
7. [Results & Performance](#7-results--performance)
8. [Technologies Used](#8-technologies-used)
9. [Reproduction](#9-reproduction)

---

## 1. Solution Overview

**What it is:** A rule-based pipeline that reads candidate profiles from a JSONL file, scores each on 6 weighted components using deterministic heuristics, and outputs a ranked top-100 submission CSV with per-candidate reasoning.

**What differentiates it from traditional keyword matching:** Traditional ATS systems count keyword overlaps (e.g., "does resume contain 'PyTorch'?"). This system goes beyond by reasoning about:

- **Skill depth, not just presence** — a skill with expert proficiency, 5 years duration, and 50 endorsements is weighted far higher than the same skill listed as "beginner" with no history
- **Title vs. substance** — a "Marketing Manager" with AI keywords is downgraded; a "Search Engineer" without explicit AI buzzwords but with a product-company trajectory is upgraded
- **Career narrative** — product-company experience is rewarded, consulting-shop background is penalized, pure-research-only profiles are filtered out
- **Behavioral realism** — available, responsive, recently-active candidates score higher regardless of profile quality
- **Search domain depth** — candidates with multiple core search/retrieval skills get a multiplier (up to 1.15×)

---

## 2. JD Understanding & Candidate Evaluation

### Key Requirements Extracted from JD

| JD Requirement | How It's Measured |
|---|---|
| Deep ML/search systems expertise | Skill Taxonomy (Tier A/B/C), Search Core skills count |
| Production vector-database experience | Tier-A skills (FAISS, Pinecone, Weaviate, Milvus, Qdrant, OpenSearch) |
| Embeddings-based retrieval | Skill Taxonomy: Embeddings, Sentence Transformers, RAG |
| Evaluation frameworks (NDCG, MRR, MAP) | Skill Taxonomy: Learning to Rank, NDCG, MRR, MAP |
| 5–9 years experience | Gaussian bell curve, plateau 4.5–9.5yr → score 1.0 |
| Product-company background (not consulting) | Career trajectory: +0.15 for product companies, −0.15 for consulting |
| Location: Pune/Noida/Hyderabad/Mumbai/Delhi NCR | Preferred city base = 1.0, other India = 0.6, outside = 0.3 |
| Available and responsive | Behavioral modifier: open-to-work flag, response rate, recency, notice period |
| Not pure-research | Pure research penalty (−0.15) if all roles are "Research" × no product × >2yr |
| Not title-hopper | Hopper penalty (≤−0.10) if avg tenure < 1.5yr |

### Candidate Signals Used

**Static profile signals:**
- Skills (name, proficiency, duration_months, endorsements)
- Current title (classified into 4 tiers)
- Career history (company, title, duration, industry per role)
- Location and country
- Years of experience

**Behavioral signals (23 fields in `redrob_signals`):**
- Profile completeness, last active date, open-to-work flag
- Recruiter response rate, interview completion rate
- Search appearance (30d), saved by recruiters (30d)
- Connection count, endorsements received
- Notice period, willing to relocate

### Beyond Keyword Matching

The core differentiator is the **trust multiplier** for skills:

```python
trust = proficiency_weight × min(duration/24, 1.0) × min(log₂(endorsements+2)/5, 1.0)
```

- Expert, 48mo, 50 endorsements → `1.0 × 1.0 × 1.0 = 1.0` (416× higher contribution)
- Beginner, 1mo, 0 endorsements → `0.3 × 0.04 × 0.2 = 0.0024`

This means a candidate with 5 well-proven skills scores higher than one with 20 superficially listed keywords.

---

## 3. Ranking Methodology

### Scoring Formula

```
raw_base = 0.40 × skill_taxonomy_score
         + 0.12 × title_fit_norm
         + 0.15 × trajectory_norm
         + 0.15 × experience_fit
         + 0.08 × location_norm
         + 0.10 × behavioral_raw

skill_penalty = 1.0 − 0.35 × exp(−n_skills / 1.5)     [if n_skills < 5]
final_score   = raw_base × skill_penalty × behavioral_modifier × search_depth_modifier
```

### Component Breakdown

#### 1. Skill Taxonomy (40% weight)

Three skill tiers:

| Tier | Multiplier | Examples |
|------|-----------|---------|
| A (core) | 3× | FAISS, Pinecone, RAG, BM25, Weaviate, Milvus, Qdrant, OpenSearch, Vector Search, Semantic Search, Embeddings, Information Retrieval, Learning to Rank |
| B (general ML) | 2× | PyTorch, TensorFlow, scikit-learn, Deep Learning, NLP, Python, MLOps |
| C (frameworks/tools) | 1× | LangChain, LlamaIndex, Elasticsearch, Prompt Engineering, Kubeflow |

Per-skill contribution: `tier_mult × trust_mult`. Aggregated via logistic saturation: `raw / (raw + 8.0)` to compress diminishing returns.

#### 2. Title Fit (12% weight)

| Tier | Score | Examples |
|------|-------|---------|
| Tier 1 | +0.40 | ML Engineer, AI Engineer, Search Engineer, NLP Engineer, Recommendation Systems Engineer |
| Tier 2 | +0.25 | Data Scientist, Software Engineer, Backend Engineer, Data Engineer |
| Tier 3 | +0.10 | Project Manager, QA Engineer, Product Manager, Engineering Manager |
| Tier 4 | −0.30 | Accountant, HR Manager, Sales Executive, Graphic Designer |
| No match | −0.15 | Unrecognized titles |

Includes alias normalization (e.g., "senior machine learning engineer" → "ML Engineer").

#### 3. Career Trajectory (15% weight)

- **Company bonus:** Product companies (Swiggy, Zomato, Google, etc.) → +0.15; Consulting firms (TCS, Infosys, etc.) → −0.15
- **Trajectory bonus:** +0.10 if ever held a Tier-1 title
- **Hopper penalty:** Up to −0.10 if avg tenure < 1.5 years
- **Pure-research penalty:** −0.15 if all career roles are "Research" with no product-company exposure and >2yr total
- **Headline boost:** +0.025 to +0.05 for search/ranking/recommendation keywords in headline

#### 4. Experience Fit (15% weight)

Gaussian centered at μ=7yr, σ=2.5yr:
- < 2yr → 0.0
- 4.5–9.5yr → 1.0 (plateau)
- > 15yr → 0.1
- Otherwise: `exp(−0.5 × ((yrs − 7) / 2.5)²)`

#### 5. Location Fit (8% weight)

- India preferred city (Pune, Noida, Delhi, Mumbai, Hyderabad, Bangalore, Gurgaon) → 1.0
- India other → 0.6
- Outside India → 0.3
- +0.2 if willing to relocate
- Minimum floor of 0.8 if candidate has ≥3 search core skills

#### 6. Behavioral Raw (10% weight)

Weighted sum of 7 signals:

| Signal | Weight | Normalization |
|---|---|---|
| Profile completeness | 0.20 | value / 100 |
| Search appearance (30d) | 0.15 | min(value / P99_cap, 1.0) |
| Saved by recruiters (30d) | 0.15 | min(value / P99_cap, 1.0) |
| Interview completion rate | 0.15 | direct (0–1) |
| Recruiter response rate | 0.15 | min(value / 0.5, 1.0) |
| Connection count | 0.10 | min(value / P99_cap, 1.0) |
| Endorsements received | 0.10 | min(value / P99_cap, 1.0) |

**P99 caps** are computed from non-honeypot candidates only to prevent synthetic outliers from inflating denominators.

#### Multipliers

**Behavioral Modifier** (0.3–1.0×): Multiplicative penalty starting at 1.0:
- Not open to work → −0.25
- Recruiter response rate < 50% → up to −0.15
- Stale activity (not logged in recently) → up to −0.15
- Notice period > 30 days → up to −0.10
- Interview completion rate < 100% → up to −0.10

**Search Depth Modifier** (1.0–1.15×): +0.05 per search core skill (FAISS, Pinecone, Weaviate, Milvus, Qdrant, OpenSearch, BM25, Learning to Rank, Vector Search, Semantic Search, Information Retrieval, NDCG, MRR, MAP, Dense Retrieval, Hybrid Search, Cross-Encoder, Re-ranking). Capped at 1.15×.

**Skill Penalty** (<5 taxonomy-relevant skills): `1.0 − 0.35 × exp(−n / 1.5)`. At 0 skills → 0.65×, at 3 skills → 0.90×, at 5+ skills → no penalty.

---

## 4. Explainability & Data Validation

### Explainability

Every top-100 candidate gets a full debug section in `logs/ranking_report.md`:

- **Component Breakdown:** Raw value, normalized value, weight, and contribution for each of the 6 components
- **Intermediate Steps:** Raw base, skill penalty, base after penalty, behavioral modifier, search depth modifier, final score
- **Skill Taxonomy Detail:** Per-skill tier, proficiency, duration, endorsements, trust multiplier, and contribution
- **Title Fit Detail:** Classification tier, raw score, normalized value
- **Career Trajectory Detail:** Each role with company bonus, Tier-1 status, all sub-scores (best company, trajectory bonus, hopper penalty, pure-research penalty, headline boost)
- **Experience Fit Detail:** Years, zone classification, score
- **Location Detail:** Country/city classification, relocation bonus, raw and normalized scores
- **Behavioral Raw Detail:** Per-signal raw value, denominator, normalized value, weight, contribution
- **Behavioral Modifier Detail:** Each penalty with formula and amount
- **Reasoning:** Plain-text summary generated from template-based interpolation

### Preventing Unsupported Justifications

Reasoning strings are generated from 4 rotating templates — each interpolates only fields that were directly computed during scoring. No LLM, no generative text, no hallucination risk.

### Handling Low-Quality / Suspicious Profiles

**Honeypot detection (3 gates):**
1. YOE < 0 or > 50 → impossible
2. Expert proficiency skill with 0 duration → contradictory
3. |years_of_experience − career_history_total| > 5 → inconsistent self-reporting

**Honeypots are:**
- Excluded from P99 behavioral cap computation (prevent outlier inflation)
- Assigned `final_score = −inf` (guaranteed to never rank)

**Low-quality profiles** are naturally suppressed:
- Few relevant skills → skill penalty (up to −35%)
- Low profile completeness → lower behavioral raw score
- Stale/inactive → behavioral modifier penalty
- Long notice period → behavioral modifier penalty

---

## 5. End-to-End Workflow

```
candidates.jsonl (100K profiles)
        │
        ▼
┌─────────────────────────────┐
│ Phase 1: Pre-scan           │
│                             │
│ For each candidate:         │
│   is_honeypot(c)?           │
│     → skip if honeypot      │
│     → collect 4 signals     │
│       from non-honeypots    │
│                             │
│ Compute P99 caps:           │
│   search_appearance_30d     │
│   saved_by_recruiters_30d   │
│   connection_count          │
│   endorsements_received     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Phase 2: Feature Extraction │
│                             │
│ For each candidate:         │
│   skill_taxonomy_score      │
│   title_fit_raw             │
│   career_trajectory_raw     │
│   experience_fit            │
│   location_raw              │
│   behavioral_raw            │
│   behavioral_modifier       │
│   n_skills (taxonomy)       │
│   search_depth (core count) │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Phase 3: Scoring            │
│                             │
│ Normalize:                  │
│   title: [-0.30, 0.40]→[0,1]│
│   trajectory: [-0.30,0.30]→[0,1]│
│   location: [0.3, 1.2]→[0,1]│
│                             │
│ For each candidate:         │
│   base = Σ(comp × weight)   │
│   if n_skills < 5:          │
│     base *= skill_penalty   │
│   final = base × modifier   │
│                × sdm        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Phase 4: Select Top 100     │
│                             │
│ argsort(final_scores) desc  │
│ take indices [:100]         │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Phase 5: Output             │
│                             │
│ Generate reasoning (4       │
│ template types, rotating)   │
│                             │
│ Write logs/ranking_report.md│
│   (full per-candidate debug)│
│                             │
│ Write submission_loc_emb_   │
│   2.csv                     │
│   (candidate_id, rank,      │
│    score, reasoning)        │
└─────────────────────────────┘
```

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   rank.py (orchestrator)             │
│                                                     │
│  load_candidates() → reads candidates.jsonl         │
│                                                     │
│  Phase 1: Pre-scan                                  │
│  ├── is_honeypot() per candidate                    │
│  ├── signal_collector for non-honeypots             │
│  └── numpy.percentile(vals, 99) → behavioral_caps   │
│                                                     │
│  Phase 2: Feature extraction                         │
│  ├── compute_taxonomy_score()                       │
│  ├── compute_title_fit_raw()                        │
│  ├── compute_career_trajectory_raw()                │
│  ├── compute_experience_fit()                       │
│  ├── compute_location_raw()                         │
│  ├── compute_behavioral_raw()                       │
│  ├── compute_behavioral_modifier()                  │
│  └── search_depth_count()                           │
│                                                     │
│  Phase 3: Scoring                                   │
│  ├── normalize() vectorized                         │
│  ├── raw_base → apply penalties → modifiers         │
│  └── final_score per candidate                      │
│                                                     │
│  Phase 4: Selection                                  │
│  └── numpy.argsort → top 100                        │
│                                                     │
│  Phase 5: Output                                    │
│  ├── generate_reasoning()                           │
│  ├── logs/ranking_report.md                         │
│  └── submission_loc_emb_2.csv                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              features.py (stateless library)         │
│                                                     │
│  Constants:                                         │
│  ├── TIER_A / TIER_B / TIER_C (skill taxonomy)      │
│  ├── PRODUCT_COMPANIES / CONSULTING_COMPANIES       │
│  ├── PREFERRED_LOCATIONS                            │
│  ├── TIER_1/2/3/4_TITLES + TITLE_ALIASES            │
│  └── ANTI_SKILLS                                    │
│                                                     │
│  Functions:                                         │
│  ├── is_honeypot()                                  │
│  ├── trust_mult()                                   │
│  ├── compute_taxonomy_score()                       │
│  ├── classify_title() / title_raw_score()           │
│  ├── compute_career_trajectory_raw()                │
│  ├── compute_experience_fit()                       │
│  ├── compute_location_raw()                         │
│  ├── compute_behavioral_raw()                       │
│  ├── compute_behavioral_modifier()                  │
│  ├── normalize()                                    │
│  ├── generate_reasoning()                           │
│  └── debug_*() (taxonomy, title, trajectory, etc.)  │
└─────────────────────────────────────────────────────┘
```

**Dependency graph:**
- `rank.py` imports from `features.py` (no circular dependencies)
- `rank.py` depends on `numpy` (argsort, percentile)
- `features.py` depends only on Python stdlib (`math`, `datetime`)
- No database, no network, no external services

---

## 7. Results & Performance

### Runtime

| Phase | Time (100K candidates) |
|---|---|
| Load JSONL | ~0.3s |
| Pre-scan + P99 caps | ~0.5s |
| Feature extraction | ~1.8s |
| Scoring + normalization | ~0.8s |
| Sort + select top 100 | ~0.1s |
| Report generation + CSV | ~1.2s |
| **Total** | **~4.7s** |

Hardware: MacBook Pro M4, 10 CPU cores, 16GB RAM. No GPU, no pre-computation, no network.

### Score Distribution (Top 100)

| Range | Count |
|---|---|
| 0.900–0.950 | 6 |
| 0.850–0.900 | 32 |
| 0.800–0.850 | 62 |

### Ranking Quality Indicators

- **Top slot** (CAND_0018499): Google + Flipkart + Zomato career history, Senior MLE title, 13 taxonomy-relevant skills, 6 search core skills, P99-range behavioral signals, no penalties applied
- **Top 10** profiles average: 6.5yr YOE, Tier-1 titles, product-company backgrounds, strong skill taxonomy scores (0.72–0.78), responsive and recently active
- **No consulting-only** candidates in top 50 — the trajectory penalty correctly deprioritizes TCS/Infosys/Wipro backgrounds
- **Outside-India candidates** are present (Berlin, Toronto) but ranked lower due to location base; only exceptional skill/trajectory scores can compensate
- **Honeypot removal** shifts behavioral caps slightly downward (2–5%) by excluding synthetic outliers

### Validation

`validate_submission.py` confirms:
- Correct header format
- Exactly 100 data rows
- Valid candidate_id format (CAND_XXXXXXX)
- Unique ranks 1–100
- Non-increasing scores by rank
- Correct tie-breaking (ascending candidate_id for equal scores)

---

## 8. Technologies Used

| Technology | Purpose | Why Selected |
|---|---|---|
| **Python 3.11** | Runtime | Required by challenge; stdlib handles JSON, CSV, math |
| **NumPy** | Vector operations | `argsort` for ranking, `percentile` for P99 caps, vectorized arrays to avoid Python loops over 100K candidates. Without numpy: 5–10× slower |
| **No ML frameworks** | Deliberate omission | Challenge JD warns against keyword-counting traps. Taxonomy + trust multipliers capture skill depth without embeddings |
| **No LLMs** | Deliberate omission | Reasoning uses template interpolation from computed fields — zero hallucination risk, no API cost, fully deterministic |
| **No database** | Deliberate omission | Single-file pipeline, no network, no setup beyond `pip install numpy` |

### Why No Embeddings or LLMs

The JD itself provides the rationale: *"The 'right answer' is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

A taxonomy-based approach is:
- **Faster** — ~5s vs minutes with embedding inference
- **Deterministic** — same input always produces same output
- **Explainable** — every score point traces to a human-readable rule
- **Network-free** — no API calls, no model downloads
- **Reproducible** — no model versioning, no stochasticity

---

## 9. Reproduction

### Prerequisites

```bash
# Python 3.11+ with numpy
pip install numpy
```

### Run

```bash
python rank.py
```

This generates:
- `logs/ranking_report.md` — full per-candidate debug report for top 100
- `submission_loc_emb_2.csv` — challenge submission file

### Validate

```bash
python validate_submission.py submission_loc_emb_2.csv
```
