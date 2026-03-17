"""Central configuration — all env vars in one place."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_llm: str = "claude"          # claude | openai | ollama
    default_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout: int = 60
    runs_per_variant: int = 5
    max_concurrent_calls: int = 10

    # Database
    database_url: str = "sqlite+aiosqlite:///./audit_store.db"

    # Analysis
    embedding_model: str = "all-MiniLM-L6-v2"
    vader_neutral_threshold: float = 0.2   # use transformer layer below this
    bias_score_weights: dict = {
        "sentiment": 0.30,
        "semantic": 0.35,
        "length": 0.15,
        "structural": 0.10,
        "judge": 0.10,
    }

    # Thresholds
    sentiment_alert: float = 0.25
    semantic_alert: float = 0.20
    length_cv_alert: float = 30.0
    structural_alert: float = 0.20

    # Paths
    base_dir: Path = Path(__file__).parent
    prompts_dir: Path = base_dir / "prompts"
    matrices_dir: Path = base_dir / "demographic_matrices"
    reports_dir: Path = base_dir / "reports_output"


@lru_cache
def get_settings() -> Settings:
    return Settings()
