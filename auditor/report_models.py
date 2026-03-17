"""Pydantic data models — the contract between every component."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    CONCERN = "concern"
    FAIL = "fail"


class BiasSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class DemographicContext(BaseModel):
    dimension: str
    value: str


class VariantPrompt(BaseModel):
    rendered: str
    context: dict[str, str]   # {dimension: value}


class LLMResponse(BaseModel):
    variant_prompt: str
    context: dict[str, str]
    response_text: str
    latency_ms: float
    token_count: int
    run_number: int
    model: str


class SentimentResult(BaseModel):
    group: str
    dimension: str
    scores: list[float]
    mean: float
    std: float


class SentimentAnalysis(BaseModel):
    results_by_group: dict[str, SentimentResult]
    f_statistic: float
    p_value: float
    cohens_d: float
    is_significant: bool
    bias_score: float   # 0-100


class SemanticSimilarityAnalysis(BaseModel):
    within_group_mean: float
    between_group_mean: float
    similarity_gap: float
    most_divergent_pair: tuple[str, str] | None
    most_divergent_score: float
    bias_score: float


class StructuralQualityResult(BaseModel):
    group: str
    word_count_mean: float
    sentence_count_mean: float
    specificity_mean: float
    completeness_mean: float
    vocabulary_complexity_mean: float


class StructuralQualityAnalysis(BaseModel):
    results_by_group: dict[str, StructuralQualityResult]
    quality_range: float
    f_statistic: float
    p_value: float
    cohens_d: float
    is_significant: bool
    bias_score: float


class JudgeComparison(BaseModel):
    response_a_id: str
    response_b_id: str
    response_a_quality: float
    response_b_quality: float
    quality_equivalent: bool
    tone_difference: BiasSeverity
    tone_description: str | None
    substance_difference: BiasSeverity
    substance_description: str | None
    assumptions_difference: BiasSeverity
    overall_bias_signal: BiasSeverity
    primary_concern: str | None


class JudgeAnalysis(BaseModel):
    comparisons: list[JudgeComparison]
    mean_bias_severity: float   # none=0, mild=25, moderate=60, severe=100
    bias_score: float


class DimensionBiasResult(BaseModel):
    demographic_dimension: str
    sentiment: SentimentAnalysis
    semantic: SemanticSimilarityAnalysis
    structural: StructuralQualityAnalysis
    judge: JudgeAnalysis
    composite_score: float
    verdict: Verdict


class RemediationRecommendation(BaseModel):
    change_type: str   # add_instruction | remove_phrase | reorder_context | reframe_task
    current_text: str | None
    suggested_text: str
    expected_impact: str
    confidence: str   # high | medium | low


class RemediationReport(BaseModel):
    root_cause_hypothesis: str
    recommendations: list[RemediationRecommendation]
    estimated_bias_reduction: str


class BiasReport(BaseModel):
    job_id: str
    prompt_template: str
    llm_name: str
    model_version: str
    demographic_matrix_name: str
    runs_per_variant: int
    total_responses: int
    created_at: datetime
    completed_at: datetime | None
    dimension_results: list[DimensionBiasResult]
    overall_score: float
    overall_verdict: Verdict
    executive_summary: str | None
    remediation: RemediationReport | None
    regulatory_docs: dict[str, str] | None
