"""Report download endpoint."""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.connection import get_session
from database.models import AuditJob
from reporting.generator import generate_pdf_report

router = APIRouter()


@router.get("/{job_id}/pdf")
async def download_pdf(job_id: str, session: AsyncSession = Depends(get_session)):
    stmt = select(AuditJob).where(AuditJob.id == job_id)
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()
    if not job or not job.report_json:
        raise HTTPException(404, "Report not available")
    from auditor.report_models import BiasReport
    report = BiasReport.model_validate_json(job.report_json)
    pdf_path = await generate_pdf_report(report)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"audit_{job_id[:8]}.pdf")
