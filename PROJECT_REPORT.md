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
