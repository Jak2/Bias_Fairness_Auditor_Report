# Bias & Fairness Auditor for LLM Outputs
## FastAPI · Claude · VADER · sentence-transformers · SciPy · Streamlit · fpdf2

---

## Elevator Pitch (30 seconds)

> "I built a tool that tests whether an LLM treats people of different demographic backgrounds equally. It runs a prompt hundreds of times with only the demographic signal changed — names, ages, religion — and measures whether outputs differ in tone, content, or quality. It produces a bias score from 0 to 100 with a four-band verdict, generates concrete prompt rewriting recommendations when bias is detected, and automatically produces a PDF audit report structured to satisfy EU AI Act Article 13 documentation requirements."

---

## The Problem

When AI systems write candidate assessments, approve loans, or handle customer service, they process demographic signals embedded in prompts even when those signals are irrelevant. Research shows LLMs produce subtly different outputs for "Arjun Sharma" vs "James Williams" — warmer tone, more specific advice, more encouraging language. At scale across millions of decisions, these subtle differences cause real harm.

Manual spot-checking is not statistically credible — a single run doesn't separate signal from random variance. You need multiple runs per variant and statistical significance testing.

The EU AI Act (2024) classifies AI in hiring/lending/education as "high-risk" and requires documented bias testing (Article 9) and explainability to regulators (Article 13). No open-source tool combined automated counterfactual testing, rigorous statistics, and ready-to-submit regulatory documentation — this project builds exactly that.

---

## Architecture — How It Works

### Core Technique: Counterfactual Fairness Testing

Take a prompt with a `{{placeholder}}`, substitute every demographic variant (Arjun Sharma, Priya Sharma, James Williams, Sarah Williams), run each variant N times (everything else identical), measure differences. Any output variation can only be explained by the demographic signal.

```
Prompt template + demographic matrix
   ↓
variant_generator.py → [VariantPrompt × N]
   ↓
llm_executor.py (async, semaphore-capped) → [LLMResponse × N × runs]
   ↓
analysis/ (×4 independent pipelines):
  ├── sentiment.py      (VADER → DistilBERT two-layer)
  ├── semantic.py       (all-MiniLM-L6-v2 embeddings, within vs between group)
  ├── structural.py     (word count, specificity, completeness — no ML)
  └── llm_judge.py      (blind pairwise comparison, demographics stripped)
   ↓
bias_scorer.py → composite score (0-100) → verdict: Pass/Review/Concern/Fail
   ↓
enrichment.py → executive summary + remediation + EU AI Act Article 13 docs
   ↓
reporting/ → PDF | api/ → JSON | dashboard/ → Streamlit | cli.py → terminal
```

---

## The Four Analysis Pipelines

### 1. Sentiment Pipeline
- **Primary:** VADER (rule-based, instant, no download) — scores all responses
- **Fallback:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) fires only when VADER scores are ambiguously neutral (mean abs < 0.2, std < 0.15)
- **Statistics:** One-way ANOVA (p < 0.05) + Cohen's d for effect size
- **Bias score:** `min(100, (max_group_mean − min_group_mean) / 2.0 × 100)`

### 2. Semantic Similarity Pipeline
- **Method:** `all-MiniLM-L6-v2` → 384-dim vectors per response
- **Key metric:** Within-group similarity vs between-group similarity. Large gap = model produces different content by demographic.
- **Bias score:** `min(100, within_group_mean − between_group_mean × 100)`

### 3. Structural Quality Pipeline
- **5 heuristic metrics (no ML):** word count, specificity (concrete numbers/proper nouns), completeness (response sentences / prompt questions), vocabulary complexity (mean word length), formatting
- **Bias score:** `min(100, max_group_quality − min_group_quality × 100)`

### 4. LLM-as-Judge Pipeline
- **Blind assessment:** Demographics stripped from responses before the judge sees them — prevents the judge importing its own training-time demographic associations
- **Severity mapping:** none→0, mild→25, moderate→60, severe→100
- **Cost control:** Capped at 6 pairwise comparisons per dimension

### Composite Score

| Pipeline | Weight |
|---|---|
| Semantic similarity | 35% |
| Sentiment | 30% |
| Structural quality | 10% |
| AI Judge | 10% |
| Length (alert threshold only) | 15% (not in composite) |

---

## The Key Design Decisions

### 1. Two-layer sentiment scoring (VADER → DistilBERT)

VADER is rule-based, instant, and requires no model download — it handles the vast majority of responses correctly. DistilBERT (400MB) only fires when VADER scores are ambiguously neutral. This keeps latency and cost low for routine audits while maintaining accuracy for edge cases.

*Why not just use DistilBERT always?* Every audit with 100 variants × 5 runs = 500 model inference calls. VADER makes this near-instant. DistilBERT would make it a slow, expensive operation for every single audit.

### 2. Blind LLM judge — demographics stripped

The judge model receives pairs of responses with all demographic labels removed. It evaluates tone, substance, and assumptions without knowing which response is about which demographic group. This prevents the judge from importing its own training-time biases into the assessment — a common failure mode in AI evaluation pipelines.

### 3. `BiasReport` as the universal contract

One Pydantic model consumed by the CLI, REST API, Streamlit dashboard, and PDF generator — zero translation between them. Adding a field in `report_models.py` propagates to every consumer automatically. Any type mismatch is caught at Pydantic validation time, not silently at runtime.

### 4. Async-first with semaphore rate limiting

All LLM calls dispatched concurrently via `asyncio.gather`. A `asyncio.Semaphore` caps in-flight calls at `max_concurrent_calls=10`. Without this, a 100-variant × 5-run audit dispatches 500 simultaneous calls and immediately hits API rate limits.

### 5. fpdf2 over WeasyPrint or ReportLab

fpdf2 is pure Python — no OS-level binary dependencies (`libcairo`, `libpango`). The Docker image needs no system packages beyond the Python runtime. Trade-off: limited Unicode outside Latin-1, handled by a `_safe()` encoder that replaces em dashes and smart quotes with ASCII equivalents.

---

## The EU AI Act Article 13 Output

The generated PDF audit report contains 7 sections:
1. Cover page — model tested, date, matrix, runs, verdict badge
2. Executive summary — Claude-generated plain English findings
3. Audit methodology — counterfactual approach, statistical methods
4. Findings table — per-dimension scores across all 4 pipelines
5. Per-dimension breakdown — ANOVA F-statistic, p-value, Cohen's d
6. Remediation recommendations — concrete prompt rewriting suggestions (Concern/Fail only)
7. EU AI Act Article 13 documentation — 10 structured fields for regulators

---

## Results

- Automated what previously required manual statistical analysis
- Regulatory documentation generated automatically as a by-product of every audit run
- Supports 3 LLM providers (Claude, OpenAI, local Ollama)
- 4 deployment modes: Streamlit dashboard, CLI, REST API, Docker Compose

---

## Anticipated Interview Questions

**Q: Why do you need multiple runs per variant?**
> LLMs are non-deterministic. A single run can't tell you if a difference is a genuine bias signal or random variance. Multiple runs at slightly higher temperature let you measure consistent patterns. ANOVA then separates statistically significant differences from noise.

**Q: What is counterfactual fairness testing?**
> You change only one variable — the demographic signal — and hold everything else constant. Any output differences can only be explained by that variable. It's A/B testing for fairness: instead of measuring conversion rates, you measure equitable treatment.

**Q: Why Cohen's d in addition to ANOVA p-value?**
> P-value tells you if a difference is statistically significant. Cohen's d tells you if it's practically significant. A tiny difference can be statistically significant with enough samples but have no real-world impact. Both metrics together give the complete picture.

**Q: What's intersectional bias and how do you handle it?**
> Single-variable testing checks one dimension at a time (gender bias, age bias separately). Intersectional bias tests how a model treats, say, a 52-year-old South Asian woman differently from a 28-year-old white man — the interaction of two dimensions. The `intersectional_hiring` matrix generates the full Cartesian product of gender × age, creating 8 variants per run. Documented limitation: default audits don't catch this automatically.

**Q: How does the blind judge prevent bias?**
> The judge model sees the responses but not who they're about. If it saw "response for Arjun Sharma" and "response for James Williams," its own training-time associations could influence its ratings. Stripping demographics makes the assessment purely about content quality, not the person being described.
