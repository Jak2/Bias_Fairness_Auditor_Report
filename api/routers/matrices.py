"""Demographic matrix CRUD."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.connection import get_session
from database.models import DemographicMatrix
from config import get_settings

router = APIRouter()
cfg = get_settings()


class CreateMatrixRequest(BaseModel):
    name: str
    description: str = ""
    variables: dict[str, list[str]]


@router.get("/")
async def list_matrices(session: AsyncSession = Depends(get_session)):
    # Load builtins from disk if not yet in DB
    await _seed_builtins(session)
    stmt = select(DemographicMatrix).order_by(DemographicMatrix.name)
    result = await session.execute(stmt)
    matrices = result.scalars().all()
    return [
        {"id": m.id, "name": m.name, "description": m.description,
         "variables": m.variables, "is_builtin": m.is_builtin}
        for m in matrices
    ]


@router.get("/{name}")
async def get_matrix(name: str, session: AsyncSession = Depends(get_session)):
    stmt = select(DemographicMatrix).where(DemographicMatrix.name == name)
    result = await session.execute(stmt)
    m = result.scalar_one_or_none()
    if not m:
        raise HTTPException(404, f"Matrix '{name}' not found")
    return {"name": m.name, "description": m.description, "variables": m.variables}


@router.post("/", status_code=201)
async def create_matrix(req: CreateMatrixRequest, session: AsyncSession = Depends(get_session)):
    m = DemographicMatrix(
        name=req.name,
        description=req.description,
        variables=req.variables,
        is_builtin=False,
        created_at=datetime.now(timezone.utc),
    )
    session.add(m)
    await session.commit()
    return {"name": m.name, "id": m.id}


async def _seed_builtins(session: AsyncSession) -> None:
    matrices_dir = cfg.matrices_dir
    if not matrices_dir.exists():
        return
    for json_file in matrices_dir.glob("*.json"):
        name = json_file.stem
        stmt = select(DemographicMatrix).where(DemographicMatrix.name == name)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if not existing:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            m = DemographicMatrix(
                name=name,
                description=data.get("description", ""),
                variables=data.get("variables", {}),
                is_builtin=True,
                created_at=datetime.now(timezone.utc),
            )
            session.add(m)
    await session.commit()
