"""FastAPI application entry point."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.connection import init_db
from api.routers import audits, matrices, reports

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Bias & Fairness Auditor API",
    description="Counterfactual fairness testing for LLM outputs. EU AI Act Article 13 compliant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audits.router, prefix="/audits", tags=["audits"])
app.include_router(matrices.router, prefix="/matrices", tags=["matrices"])
app.include_router(reports.router, prefix="/reports", tags=["reports"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "bias-fairness-auditor"}
