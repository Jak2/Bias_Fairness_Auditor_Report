# ⚖️ Bias & Fairness Auditor for LLM Outputs

A specialized tool that measures and mitigates bias in language model outputs, generating statistically rigorous evidence and regulatory-compliant documentation for pre-deployment testing.

## Features

- **Multi-Pipeline Bias Detection**: Evaluates semantic similarity, structural bias, and tonal differences.
- **Automated Validation**: Generates variant prompts via the Cartesian product of demographic matrices.
- **Statistical Rigour**: Implements ANOVA, Chi-squared, and Cohen's d tests to isolate significant fairness metrics without false positives.
- **Regulatory Reporting**: Auto-generates EU AI Act compliant PDF reports.
- **Interactive Dashboard**: Streamlit-based UI for real-time visualization, response exploration, and prompt remediation.

## Tech Stack Overview

- **Backend & Logic**: Python 3.11+, FastAPI, `asyncio`
- **NLP & Analysis**: Anthropic SDK, OpenAI SDK, VADER Sentiment, `sentence-transformers` (`all-MiniLM-L6-v2`), SciPy
- **Data & Storage**: SQLite (Local), PostgreSQL (Production), SQLAlchemy Async
- **Frontend & Reporting**: Streamlit, Plotly, `fpdf2`
- **Infrastructure**: Docker, Docker Compose

## System Requirements

- **Python**: Version 3.11 or greater
- **Hardware**: Standard modern CPU (No GPU required); 8GB RAM minimum (16GB recommended for full Docker stack)
- **Storage**: ~500MB free disk space for dependencies and model weights
- **API Keys**: Anthropic API Key (default) or OpenAI API Key

## Installation & Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit .env and securely add your ANTHROPIC_API_KEY
```

## Usage

### 1. Run the Streamlit Dashboard (Recommended)
```bash
streamlit run dashboard/app.py
```
*Access at `http://localhost:8501`. Paste a prompt template, select a demographic matrix, and click Start Audit.*

### 2. Run via Command Line Interface (CLI)
```bash
python cli.py \
  --template "Assess candidate {{candidate_name}} for a Senior Engineer role." \
  --matrix gender_names_india \
  --runs 3
```

### 3. Run the REST API
```bash
uvicorn api.main:app --reload
```
*API documentation available at `http://localhost:8000/docs`.*

### 4. Run full stack via Docker Compose
```bash
docker-compose up
```
*Spins up FastAPI, Streamlit UI, and a PostgreSQL database.*

## Project Structure

```text
bias-fairness-auditor/
├── api/                  # FastAPI complete REST API application & routers
├── auditor/              # Core execution, variants, AI judge, statistical analysis
├── dashboard/            # Streamlit multi-tab frontend and Plotly visualizations
├── database/             # SQLAlchemy ORM and database configurations
├── demographic_matrices/ # JSON baseline matrix definitions (Age, Gender, Nationality, etc.)
├── prompts/              # Internal system evaluation prompt templates
├── reporting/            # PDF generation pipelines via fpdf2
├── cli.py                # Command-line interface entry point
├── config.py             # Global configurations via Pydantic
└── requirements.txt      # Main project dependencies
```

## Author
**Jaya Arun Kumar Tulluri** — v1.0, March 2026
