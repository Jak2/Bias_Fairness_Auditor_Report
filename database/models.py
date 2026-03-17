"""SQLAlchemy async models for the audit store."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AuditJob(Base):
    __tablename__ = "audit_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prompt_template_text: Mapped[str] = mapped_column(Text)
    prompt_template_hash: Mapped[str] = mapped_column(String(64), index=True)
    llm_name: Mapped[str] = mapped_column(String(32))
    llm_model_version: Mapped[str] = mapped_column(String(64))
    demographic_matrix_name: Mapped[str] = mapped_column(String(128))
    runs_per_variant: Mapped[int] = mapped_column(Integer, default=5)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|completed|failed
    created_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    overall_bias_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    total_responses: Mapped[int] = mapped_column(Integer, default=0)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # full BiasReport JSON


class DemographicMatrix(Base):
    __tablename__ = "demographic_matrices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    variables: Mapped[dict] = mapped_column(JSON)  # {dimension: [values]}
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class RemediationHistory(Base):
    __tablename__ = "remediation_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audit_job_id: Mapped[str] = mapped_column(String(36), index=True)
    recommendation_type: Mapped[str] = mapped_column(String(32))
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_text: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(16), default="pending")  # accepted|rejected|pending
    outcome_bias_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
