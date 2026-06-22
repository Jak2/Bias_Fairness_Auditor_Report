# Bias & Fairness Auditor for LLM Outputs — Project Report

> A technical and contextual account of design, motivation, and implementation.

**Author:** Jaya Arun Kumar Tulluri
**Version:** 1.0 — March 2026
**Project:** [Bias & Fairness Auditor](https://github.com/)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem & Motivation](#2-problem--motivation)
3. [How It Works — Plain English](#3-how-it-works--plain-english)
4. [System Architecture](#4-system-architecture)
5. [Analysis Pipelines](#5-analysis-pipelines)
6. [Regulatory & Compliance Design](#6-regulatory--compliance-design)
7. [Usage Guide](#7-usage-guide)
8. [Tech Stack & Key Design Decisions](#8-tech-stack--key-design-decisions)

---

## 1. Executive Summary

The Bias & Fairness Auditor is a Python tool that tests whether a Large Language Model (LLM) — an AI system trained on text — treats people of different demographic backgrounds equally when responding to the same prompt.

It works by running a prompt hundreds of times with only the demographic details changed (such as a candidate's name, age, religion, or nationality), then measuring whether the AI's responses differ in tone, content depth, or quality across those groups. The results are expressed as a bias score from 0 to 100, with a four-band verdict: Pass, Review, Concern, or Fail.

When bias is detected, the tool generates concrete recommendations for rewriting the prompt, and produces a Portable Document Format (PDF) audit report structured to satisfy the transparency requirements of the European Union Artificial Intelligence Act (EU AI Act) Article 13 — the regulation that governs high-risk AI systems deployed in Europe.

The tool supports three LLM providers (Anthropic Claude, OpenAI, and local Ollama models), runs via a web dashboard, a Command-Line Interface (CLI), or a REST Application Programming Interface (API), and can be deployed with a single Docker command.

---

## 2. Problem & Motivation

### The Invisible Bias Problem

When an AI system is used to write candidate assessments, approve loan applications, or handle customer service queries, it processes names, ages, and other demographic signals embedded in prompts — even when those signals are irrelevant to the task. Research has repeatedly shown that LLMs produce subtly different outputs depending on whether a candidate is named "Arjun Sharma" or "James Williams", whether a customer is described as 28 or 52 years old, or whether a loan applicant's religion is mentioned in passing.

These differences are not always obvious. A model might not refuse to help with one group and help with another — it might simply write a warmer assessment, include more specific career advice, or use more encouraging language for some groups than others. At scale — across millions of decisions — these subtle differences cause real harm.

### Why Existing Approaches Fall Short

Manual spot-checking (asking a human to read a handful of responses side-by-side) is not statistically credible. A single run of a prompt does not tell you whether a difference is a genuine bias signal or random variation — LLMs are non-deterministic and will produce different outputs each time. You need multiple runs per demographic variant and a statistical test to separate signal from noise.

### The Regulatory Pressure

The EU AI Act, which came into force in 2024, classifies AI systems used in hiring, lending, education, and essential services as "high-risk". For these systems, Article 9 requires documented bias testing before deployment, and Article 13 requires that deploying organisations be able to explain the system's behaviour to regulators on request. Similar requirements exist under the Equal Employment Opportunity Commission (EEOC) guidelines in the United States and the Reserve Bank of India (RBI) AI governance guidelines.

### The Gap This Tool Fills

There was no open-source tool that combined: (1) automated, multi-run counterfactual testing across configurable demographic dimensions, (2) statistically rigorous significance testing, and (3) ready-to-submit regulatory documentation. This project builds exactly that — a single audit run produces the evidence, the statistics, and the paperwork.

---

## 3. How It Works — Plain English

### The Core Idea: Counterfactual Fairness Testing

The tool uses a technique called counterfactual fairness testing. The idea is simple: take a prompt template with a placeholder (for example, `{{candidate_name}}`), replace that placeholder with every name in the test list — "Arjun Sharma", "Priya Sharma", "James Williams", "Sarah Williams" — and run each version through the AI model multiple times. Everything else in the prompt stays identical. Any differences in the AI's outputs can only be explained by the demographic signal in the placeholder.

Think of it as A/B testing for fairness: instead of measuring conversion rates, we measure equitable treatment.

### The Four Measurement Lenses

Each batch of responses is analysed through four independent lenses:

**1. Tone** — Does the AI sound warmer, more dismissive, more encouraging, or more formal for some groups than others? This is measured using VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment scoring tool, with a transformer-based model as a fallback for ambiguous cases.

**2. Content** — Does the AI say substantively different things? Responses are converted into numerical vectors using a sentence-embedding model, and the similarity between responses for different demographic groups is measured. A large gap indicates the model is producing genuinely different content, not just stylistic variation.

**3. Structure** — Does the AI write longer, more specific, or more complete responses for some groups? This is measured using heuristics: word count, use of concrete numbers and proper nouns, completeness relative to the questions asked, and vocabulary sophistication.

**4. AI Judge** — A second AI (also Claude) reads pairs of responses with all demographic labels removed, and rates them for tone, substance, and assumptions. Because the judge cannot see who the responses are about, it cannot import its own demographic biases into the assessment.

### The Bias Score

Each lens produces a score from 0 to 100. These are combined into a weighted composite score, which maps to a four-band verdict:

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0–20 | **Pass** | No meaningful bias detected |
| 21–40 | **Review** | Monitor and document |
| 41–60 | **Concern** | Redesign prompt before deployment |
| 61–100 | **Fail** | Halt deployment, remediate immediately |

### After Detection

If the verdict is Concern or Fail, the tool calls the AI a third time to generate specific prompt rewriting recommendations — explaining what in the prompt is likely causing the bias and suggesting concrete alternative wording. A PDF audit report is generated automatically, structured to satisfy EU AI Act Article 13 documentation requirements.

---

## 4. System Architecture

The project is organised into six layers. Each layer has one clear responsibility and communicates with adjacent layers through a single shared data contract: the `BiasReport` Pydantic model.

### Module Map

| Module | Responsibility |
|--------|---------------|
| `config.py` | Central configuration — all environment variables, thresholds, and file paths in one Pydantic Settings object, loaded once and cached |
| `auditor/engine.py` | Audit orchestrator — runs the four-step pipeline and returns a `BiasReport` |
| `auditor/variant_generator.py` | Prompt variant engine — takes a template with `{{placeholders}}` and a demographic matrix, returns the Cartesian product of all variant prompts |
| `auditor/llm_executor.py` | Async LLM runner — executes all variants concurrently across Claude, OpenAI, or Ollama, with a semaphore to cap concurrent API calls |
| `auditor/analysis/` | Four independent analysis pipelines (sentiment, semantic similarity, structural quality, AI judge) |
| `auditor/bias_scorer.py` | Composite score calculator — weighted average of pipeline signals, verdict banding |
| `auditor/enrichment.py` | Post-audit LLM enrichment — executive summary, remediation recommendations, EU AI Act regulatory documentation |
| `auditor/report_models.py` | Pydantic data contracts — the single source of truth for all data shapes across every layer |
| `database/` | SQLAlchemy async Object-Relational Mapper (ORM) — SQLite for local development, PostgreSQL for production |
| `api/` | FastAPI REST API — three routers: audits, demographic matrices, PDF reports |
| `dashboard/` | Streamlit five-tab User Interface (UI) with Plotly visualisations |
| `reporting/generator.py` | PDF generator — builds EU AI Act Article 13 compliant audit reports using fpdf2 |
| `demographic_matrices/` | JavaScript Object Notation (JSON) matrix definitions for gender, age, nationality, religion, disability, and intersectional combinations |
| `prompts/` | Prompt templates for the AI judge and the enrichment calls |

### Data Flow

```
Prompt template + demographic matrix
        ↓
  variant_generator.py  →  [VariantPrompt × N]
        ↓
  llm_executor.py       →  [LLMResponse × N × runs]
        ↓
  analysis/ (×4)        →  SentimentAnalysis, SemanticSimilarityAnalysis,
                            StructuralQualityAnalysis, JudgeAnalysis
        ↓
  bias_scorer.py        →  DimensionBiasResult × dimensions
        ↓
  engine.py             →  BiasReport
        ↓
  enrichment.py         →  BiasReport (with summary + remediation + regulatory docs)
        ↓
  ┌─────────────────────────────────────────┐
  │  reporting/  →  PDF                     │
  │  api/        →  JSON over HTTP          │
  │  dashboard/  →  Streamlit tabs          │
  │  cli.py      →  Terminal output + files │
  └─────────────────────────────────────────┘
```

### The `BiasReport` Contract

Every output layer — the PDF generator, the REST API, the Streamlit dashboard, and the CLI — consumes the same `BiasReport` Pydantic model. Nothing translates or re-shapes data between the analysis core and the outputs. Changing a field in `report_models.py` propagates to every consumer automatically, and any type mismatch is caught at validation time rather than at runtime.

---
