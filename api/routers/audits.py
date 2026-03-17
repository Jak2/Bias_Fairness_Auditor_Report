"""Audit job CRUD endpoints."""
from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.connection import get_session
from database.models import AuditJob
from auditor.engine import run_audit
from auditor.report_models import BiasReport

log = logging.getLogger(__name__)
router = APIRouter()


class CreateAuditRequest(BaseModel):
    prompt_template: str
    demographic_matrix: dict[str, list[str]]
    matrix_name: str = "custom"
    llm: str = "claude"
    model: str | None = None
    runs_per_variant: int = 5
    enable_judge: bool = True


async def _run_and_persist(job_id: str, req: CreateAuditRequest):
    from database.connection import AsyncSessionLocal
    try:
        report = await run_audit(
            prompt_template=req.prompt_template,
            demographic_matrix=req.demographic_matrix,
            matrix_name=req.matrix_name,
            llm=req.llm,
            model=req.model,
            runs_per_variant=req.runs_per_variant,
            enable_judge=req.enable_judge,
            job_id=job_id,
        )
        async with AsyncSessionLocal() as session:
            stmt = select(AuditJob).where(AuditJob.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.completed_at = report.completed_at
                job.overall_bias_score = report.overall_score
                job.overall_verdict = report.overall_verdict.value
                job.total_responses = report.total_responses
                job.report_json = report.model_dump_json()
                await session.commit()
    except Exception as exc:
        log.exception("Audit %s failed: %s", job_id, exc)
        from database.connection import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            stmt = select(AuditJob).where(AuditJob.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                await session.commit()


@router.post("/", status_code=202)
async def create_audit(
    req: CreateAuditRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    import uuid
    job_id = str(uuid.uuid4())
    tmpl_hash = hashlib.sha256(req.prompt_template.encode()).hexdigest()
    job = AuditJob(
        id=job_id,
        prompt_template_text=req.prompt_template,
        prompt_template_hash=tmpl_hash,
        llm_name=req.llm,
        llm_model_version=req.model or "default",
        demographic_matrix_name=req.matrix_name,
        runs_per_variant=req.runs_per_variant,
        status="running",
        created_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.commit()
    background.add_task(_run_and_persist, job_id, req)
    return {"job_id": job_id, "status": "running"}


@router.get("/{job_id}")
async def get_audit(job_id: str, session: AsyncSession = Depends(get_session)):
    stmt = select(AuditJob).where(AuditJob.id == job_id)
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Audit job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "overall_score": job.overall_bias_score,
        "verdict": job.overall_verdict,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "report": json.loads(job.report_json) if job.report_json else None,
    }


@router.get("/")
async def list_audits(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AuditJob).order_by(AuditJob.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    jobs = result.scalars().all()
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "matrix": j.demographic_matrix_name,
            "llm": j.llm_name,
            "score": j.overall_bias_score,
            "verdict": j.overall_verdict,
            "created_at": j.created_at,
        }
        for j in jobs
    ]
