"""
Audit Engine — orchestrates the complete audit job lifecycle.

Call `run_audit()` with a prompt template + matrix.
Returns a fully populated BiasReport.
"""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from auditor.bias_scorer import overall_verdict, score_dimension
from auditor.llm_executor import execute_variants
from auditor.report_models import BiasReport, Verdict
from auditor.variant_generator import generate_variants, group_by_dimension
from auditor.analysis.sentiment import analyse_sentiment
from auditor.analysis.semantic_similarity import analyse_semantic_similarity
from auditor.analysis.structural_quality import analyse_structural_quality
from auditor.analysis.llm_judge import analyse_with_judge
from config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()


async def run_audit(
    prompt_template: str,
    demographic_matrix: dict[str, list[str]],
    matrix_name: str = "custom",
    llm: str | None = None,
    model: str | None = None,
    runs_per_variant: int | None = None,
    enable_judge: bool = True,
    job_id: str | None = None,
) -> BiasReport:
    """
    Full audit pipeline:
      1. Generate variants
      2. Execute LLM calls
      3. Run 4 analysis pipelines per demographic dimension
      4. Score and render report

    Returns BiasReport — everything downstream needs.
    """
    job_id = job_id or str(uuid.uuid4())
    runs = runs_per_variant or cfg.runs_per_variant
    llm = llm or cfg.default_llm
    model = model or cfg.default_model

    log.info("Audit %s started — llm=%s model=%s", job_id, llm, model)
    started = datetime.now(timezone.utc)

    # 1. Generate variants
    variants = generate_variants(prompt_template, demographic_matrix)
    log.info("Generated %d variants", len(variants))

    # 2. Execute LLM calls
    responses = await execute_variants(variants, runs_per_variant=runs, llm=llm, model=model)
    log.info("Collected %d responses", len(responses))

    # 3. Analyse per dimension
    dimensions = list(demographic_matrix.keys())
    dimension_results = []

    for dim in dimensions:
        log.info("Analysing dimension: %s", dim)
        sentiment = analyse_sentiment(responses, dim)
        semantic = analyse_semantic_similarity(responses, dim)
        structural = analyse_structural_quality(responses, dim)

        if enable_judge and cfg.anthropic_api_key:
            judge = await analyse_with_judge(responses, dim, prompt_template)
        else:
            from auditor.report_models import JudgeAnalysis
            judge = JudgeAnalysis(comparisons=[], mean_bias_severity=0.0, bias_score=0.0)

        result = score_dimension(sentiment, semantic, structural, judge, dim)
        dimension_results.append(result)

    # 4. Overall scoring
    overall_score, verdict = overall_verdict(dimension_results)

    completed = datetime.now(timezone.utc)

    report = BiasReport(
        job_id=job_id,
        prompt_template=prompt_template,
        llm_name=llm,
        model_version=model,
        demographic_matrix_name=matrix_name,
        runs_per_variant=runs,
        total_responses=len(responses),
        created_at=started,
        completed_at=completed,
        dimension_results=dimension_results,
        overall_score=overall_score,
        overall_verdict=verdict,
        executive_summary=None,
        remediation=None,
        regulatory_docs=None,
    )

    log.info(
        "Audit %s completed — overall_score=%.1f verdict=%s",
        job_id, overall_score, verdict,
    )
    return report
