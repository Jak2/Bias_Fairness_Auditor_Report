# Design Spec — Bias & Fairness Auditor Project Report

**Date:** 2026-06-22
**Author:** Claude Code (brainstorming skill)
**Deliverable:** `PROJECT_REPORT.md` in the project root
**Approach:** Layered Briefing (Approach A) — plain English opening, progressively technical, single document, mixed audience

---

## Document Identity

- **Title:** Bias & Fairness Auditor for LLM Outputs — Project Report
- **Subtitle:** A technical and contextual account of design, motivation, and implementation
- **Author:** Jaya Arun Kumar Tulluri — v1.0, March 2026
- **Format:** Markdown (`PROJECT_REPORT.md`)
- **Tone:** Plain English first, deepening technically; no jargon left unexplained
- **Length:** ~2,500–3,500 words

---

## Section Specifications

### Section 1 — Executive Summary
**Audience:** Non-technical (compliance officer, recruiter, PM)
**Length:** 150–200 words
**Content:**
- One-paragraph plain-English description of what the tool does and the problem it solves
- Key capability highlights: automated bias detection, statistical rigour, regulatory compliance
- Verdict on what makes this project non-trivial (4 detection pipelines, EU AI Act compliance, multi-provider LLM support)
- Who would use it and in what context (pre-deployment AI testing, compliance audits, vendor comparison)

---

### Section 2 — Problem & Motivation
**Audience:** All
**Length:** 300–400 words
**Content:**
- The core problem: LLMs exhibit demographic bias that is invisible without systematic testing — a model may produce subtly different assessments of "Arjun Sharma" vs "James Williams" for the same job role
- Why this matters: hiring AI, lending AI, customer service bots — consequential decisions at scale
- The regulatory pressure: EU AI Act (Articles 9 & 13) requires documented bias testing before high-risk AI deployment; India's RBI AI guidelines; US EEOC
- Why existing approaches fall short: manual spot-checking is not statistically credible; single-run testing ignores variance
- The gap this project fills: automated, statistically rigorous, regulatory-reportable bias auditing for any LLM prompt

---

### Section 3 — How It Works (Plain English)
**Audience:** Non-technical / general
**Length:** 350–450 words
**Content:**
- Counterfactual fairness concept explained without jargon: "we run the same prompt hundreds of times, changing only the person's name/age/religion, and measure whether the AI's responses differ"
- The four measurement lenses (plain English):
  1. Tone — does the AI sound warmer or more dismissive for some groups?
  2. Content — does it say substantively different things?
  3. Structure — does it write more detailed responses for some groups?
  4. AI Judge — a second AI reads pairs of responses without knowing who's who and flags quality differences
- What a bias score means: 0–100 scale, four verdict bands (Pass / Review / Concern / Fail)
- What happens after detection: prompt remediation recommendations and a PDF report for regulators
- Analogy: "think of it like A/B testing, but for fairness — instead of conversion rates, we're measuring equitable treatment"

---

### Section 4 — System Architecture
**Audience:** Engineers / technical reviewers
**Length:** 400–500 words
**Content:**
- High-level architecture: 6 layers — Config, Auditor Core, Analysis Pipelines, Database, API, Dashboard
- Module map with one-line purpose per module:
  - `config.py` — Pydantic Settings, single source of truth for all env vars and thresholds
  - `auditor/engine.py` — orchestrates the full audit lifecycle
  - `auditor/variant_generator.py` — Cartesian product engine for prompt variants
  - `auditor/llm_executor.py` — async, semaphore-limited, multi-provider LLM runner
  - `auditor/enrichment.py` — post-audit LLM calls for summary, remediation, regulatory docs
  - `auditor/bias_scorer.py` — weighted composite score + verdict banding
  - `auditor/report_models.py` — Pydantic contracts shared across all layers
  - `auditor/analysis/` — 4 independent analysis pipeline modules
  - `database/` — SQLAlchemy async ORM, SQLite (dev) / PostgreSQL (prod)
  - `api/` — FastAPI REST with 3 routers: audits, matrices, reports
  - `dashboard/` — Streamlit 5-tab UI with Plotly visualisations
  - `reporting/generator.py` — fpdf2 PDF builder, EU AI Act compliant
  - `demographic_matrices/` — JSON baseline test matrices (gender, age, nationality, religion, disability)
  - `prompts/` — system and user prompt templates for LLM judge and enrichment
- Data flow: prompt template + matrix → variants → LLM responses → 4 analysis pipelines → composite score → BiasReport → PDF/JSON/dashboard
- Key design principle: `BiasReport` Pydantic model is the single contract that all outputs (PDF, API, dashboard, CLI) consume — nothing talks to raw data

---

### Section 5 — Analysis Pipelines
**Audience:** Engineers
**Length:** 500–600 words
**Content — one subsection per pipeline:**

**5.1 Sentiment Pipeline**
- Two-layer: VADER (rule-based, instant) → transformer fallback (DistilBERT) only when VADER scores cluster near neutral (mean < 0.2, stdev < 0.15)
- Groups responses by demographic value, computes per-group mean compound scores
- Statistical test: one-way ANOVA + Cohen's d across groups
- Bias score: normalised max sentiment gap (gap of 2.0 → score of 100)

**5.2 Semantic Similarity Pipeline**
- Embeds all responses with `sentence-transformers` `all-MiniLM-L6-v2`; fallback to character bigram TF-IDF cosine (zero extra dependencies)
- Computes within-group mean similarity and between-group mean similarity
- Bias signal: large within-group / small between-group gap = the model produces substantively different content per demographic
- Identifies most divergent group pair

**5.3 Structural Quality Pipeline**
- Pure heuristic, no ML: word count, sentence count, specificity (concrete nouns/numbers fraction), completeness (question-answer ratio heuristic), vocabulary complexity (mean word length), formatting (bullets/headers/paragraphs)
- Groups by demographic value, ANOVA + Cohen's d on composite quality score
- Bias signal: quality range across groups (max − min) normalised to 0–100

**5.4 LLM-as-Judge Pipeline**
- Blind pairwise assessment: judge sees Response A / Response B with demographic labels stripped
- Judge model (Claude) evaluates tone, substance, assumptions, language on a structured JSON schema
- Capped at 6 pairs per dimension to control API cost
- Severity mapped: none=0, mild=25, moderate=60, severe=100

**Composite Scoring**
- Weighted average: sentiment 30%, semantic 35%, length 15%, structural 10%, judge 10%
- Normalised to 0–100; verdict bands: 0–20 Pass, 21–40 Review, 41–60 Concern, 61–100 Fail
- Dimension scores averaged for overall verdict

---

### Section 6 — Regulatory & Compliance Design
**Audience:** Compliance / legal
**Length:** 250–300 words
**Content:**
- Why EU AI Act Article 13 is the target: transparency obligations for high-risk AI systems, requires documented evidence of bias testing
- What the tool generates: 10-field regulatory documentation block (system identification, intended purpose, demographic scope, audit methodology, findings summary, limitations, remediation actions, monitoring plan, human oversight requirements, contact)
- Fallback regulatory doc generation (no API key required) vs LLM-generated (richer, context-aware)
- PDF format: 7 sections, cover page with verdict badge, methodology, findings table, per-dimension breakdown, remediation, EU AI Act docs
- Limitations documented in the report itself: single-variable counterfactual testing; intersectional bias requires separate audit
- Monitoring recommendation baked into output: quarterly re-audits, alert on >10-point score increase

---

### Section 7 — Usage Guide
**Audience:** Engineers / practitioners
**Length:** 300–350 words
**Content:**
- 4 usage modes with example commands:
  1. Streamlit Dashboard (recommended for first use) — `streamlit run dashboard/app.py`
  2. CLI — full example with `--template`, `--matrix`, `--runs`, `--judge`, `--pdf` flags
  3. REST API — `uvicorn api.main:app --reload`, link to `/docs`
  4. Docker Compose — `docker-compose up`, what each service does
- Built-in demographic matrices listed with descriptions:
  - `gender_names_india` — matched-status Indian name pairs (Arjun/Priya Sharma, Rohan/Kavya Mehta)
  - `gender_names_global` — global gender name pairs
  - `age_groups` — age bracket testing
  - `nationality_global` — global nationality variants
  - `religion_india` — Indian religious context names
  - `disability_context` — disability-context prompts
  - `intersectional_hiring` — gender × age Cartesian product (8 combinations)
- Custom matrix JSON format shown
- Output artefacts: audit JSON, audit PDF, Streamlit dashboard tabs explained

---

### Section 8 — Tech Stack & Key Design Decisions
**Audience:** Portfolio / recruiters / engineers
**Length:** 300–400 words
**Content:**
- Tech stack table: layer → technology → reason chosen
- 5 non-obvious design decisions worth calling out:
  1. **Pydantic as the universal contract** — `BiasReport` model is consumed by CLI, API, dashboard, and PDF generator with zero translation layer; changing one field propagates everywhere
  2. **Async-first with semaphore rate limiting** — `asyncio.gather` across all variants × runs, capped by `max_concurrent_calls` semaphore; avoids API rate-limit errors at scale
  3. **Two-layer sentiment (VADER → transformer fallback)** — VADER is instant and handles most cases; transformer only fires when scores are ambiguously neutral, keeping cost and latency low
  4. **Blind LLM judge** — demographic labels are stripped before the judge sees responses, preventing the judge model from importing its own training biases into the assessment
  5. **fpdf2 over WeasyPrint/Reportlab** — pure Python, no OS-level binary dependencies, works in Docker without system packages; trade-off is limited Unicode support (handled by `_safe()` encoder)
- Multi-provider architecture note: Claude / OpenAI / Ollama support via a single `execute_variants()` abstraction; adding a new provider requires one new `_call_X()` function
- SQLite (dev) / PostgreSQL (prod) via SQLAlchemy async — zero code change to switch

---

## Out of Scope

- Code-level API reference (covered by FastAPI `/docs` auto-generation)
- Test suite documentation (no tests in current codebase)
- Deployment runbook beyond Docker Compose

---

## File Output

- **Path:** `/media/asterisk/1722F6694CF9F147/project/my_learning_projects/Bias_Fairness_Auditor_Report/PROJECT_REPORT.md`
- **Commit:** Yes, after writing
