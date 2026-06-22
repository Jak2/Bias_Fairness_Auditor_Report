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
