"""
Post-audit enrichment: executive summary + remediation + regulatory docs.
Call after run_audit() to fully populate the BiasReport.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from auditor.report_models import BiasReport, RemediationReport, RemediationRecommendation, Verdict
from config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()


async def _call_claude(system: str, user: str, model: str) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
    resp = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


async def generate_executive_summary(report: BiasReport) -> str:
    """Generate plain-English CCO-targeted executive summary."""
    if not cfg.anthropic_api_key:
        return _fallback_summary(report)
    try:
        template = (cfg.prompts_dir / "bias_summary_generation.txt").read_text(encoding="utf-8")
        # Split system/user
        parts = template.split("USER:")
        system_part = parts[0].replace("SYSTEM:", "").strip()
        user_template = parts[1].strip() if len(parts) > 1 else parts[0]

        worst = max(report.dimension_results, key=lambda r: r.composite_score, default=None)
        scores_summary = {dr.demographic_dimension: dr.composite_score for dr in report.dimension_results}

        user_msg = user_template.format(
            prompt_description=report.prompt_template[:200] + "...",
            llm_name=f"{report.llm_name} ({report.model_version})",
            dimensions_tested=", ".join(dr.demographic_dimension for dr in report.dimension_results),
            total_runs=report.total_responses,
            results_json=json.dumps(scores_summary, indent=2),
            bias_scores_per_dimension=json.dumps(scores_summary, indent=2),
            overall_verdict=report.overall_verdict.value.upper(),
        )
        return await _call_claude(system_part, user_msg, cfg.default_model)
    except Exception as exc:
        log.warning("Summary generation failed: %s", exc)
        return _fallback_summary(report)


def _fallback_summary(report: BiasReport) -> str:
    worst = max(report.dimension_results, key=lambda r: r.composite_score, default=None)
    return (
        f"This bias audit of '{report.llm_name}' ({report.model_version}) yielded an overall composite "
        f"bias score of {report.overall_score:.1f}/100, resulting in a verdict of "
        f"'{report.overall_verdict.value.upper()}'. "
        + (
            f"The highest-scoring dimension was '{worst.demographic_dimension}' "
            f"(score {worst.composite_score:.1f}), with sentiment ANOVA p={worst.sentiment.p_value:.4f} "
            f"and Cohen's d={worst.sentiment.cohens_d:.3f}. "
            if worst else ""
        )
        + f"Total LLM calls: {report.total_responses}. "
        f"Score interpretation: 0-20 Pass, 21-40 Review, 41-60 Concern, 61-100 Fail."
    )


async def generate_remediation(report: BiasReport) -> RemediationReport | None:
    """Generate prompt remediation recommendations for concern/fail verdicts."""
    if report.overall_verdict not in (Verdict.CONCERN, Verdict.FAIL):
        return None
    if not cfg.anthropic_api_key:
        return None
    try:
        template = (cfg.prompts_dir / "remediation_recommendations.txt").read_text(encoding="utf-8")
        parts = template.split("USER:")
        system_part = parts[0].replace("SYSTEM:", "").strip()
        user_template = parts[1].strip() if len(parts) > 1 else parts[0]

        worst = max(report.dimension_results, key=lambda r: r.composite_score)
        findings = {
            "dimension": worst.demographic_dimension,
            "composite_score": worst.composite_score,
            "sentiment_bias": worst.sentiment.bias_score,
            "semantic_bias": worst.semantic.bias_score,
            "structural_bias": worst.structural.bias_score,
        }
        bias_pattern = (
            f"Sentiment gap: {worst.sentiment.bias_score:.1f}, "
            f"Semantic gap: {worst.semantic.bias_score:.1f}, "
            f"Structural gap: {worst.structural.bias_score:.1f}"
        )
        user_msg = user_template.format(
            prompt_template=report.prompt_template,
            audit_findings_json=json.dumps(findings, indent=2),
            worst_dimension=worst.demographic_dimension,
            bias_pattern_description=bias_pattern,
        )
        raw = await _call_claude(system_part, user_msg, cfg.default_model)
        data = _parse_json(raw)
        recs = [
            RemediationRecommendation(
                change_type=r.get("change_type", "add_instruction"),
                current_text=r.get("current_text"),
                suggested_text=r.get("suggested_text", ""),
                expected_impact=r.get("expected_impact", ""),
                confidence=r.get("confidence", "medium"),
            )
            for r in data.get("recommendations", [])
        ]
        return RemediationReport(
            root_cause_hypothesis=data.get("root_cause_hypothesis", ""),
            recommendations=recs,
            estimated_bias_reduction=data.get("estimated_bias_reduction", "Unknown"),
        )
    except Exception as exc:
        log.warning("Remediation generation failed: %s", exc)
        return None


async def generate_regulatory_docs(report: BiasReport, system_description: str = "", intended_purpose: str = "", deployment_context: str = "") -> dict[str, str] | None:
    """Generate EU AI Act Article 13 documentation sections."""
    if not cfg.anthropic_api_key:
        return _fallback_regulatory_docs(report)
    try:
        template = (cfg.prompts_dir / "regulatory_documentation.txt").read_text(encoding="utf-8")
        parts = template.split("USER:")
        system_part = parts[0].replace("SYSTEM:", "").strip()
        user_template = parts[1].strip() if len(parts) > 1 else parts[0]

        user_msg = user_template.format(
            system_description=system_description or f"LLM system: {report.llm_name}",
            intended_purpose=intended_purpose or "AI-assisted decision making",
            deployment_context=deployment_context or "Enterprise deployment",
            dimensions_count=len(report.dimension_results),
            total_runs=report.total_responses,
            results_summary=f"Overall bias score: {report.overall_score:.1f}/100, verdict: {report.overall_verdict.value}",
        )
        raw = await _call_claude(system_part, user_msg, cfg.default_model)
        return _parse_json(raw)
    except Exception as exc:
        log.warning("Regulatory docs generation failed: %s", exc)
        return _fallback_regulatory_docs(report)


def _fallback_regulatory_docs(report: BiasReport) -> dict[str, str]:
    return {
        "system_identification": f"{report.llm_name} {report.model_version}",
        "intended_purpose": "To be completed by deploying organisation",
        "demographic_scope": ", ".join(dr.demographic_dimension for dr in report.dimension_results),
        "audit_methodology": f"Counterfactual fairness testing, {report.total_responses} LLM executions, ANOVA + Cohen's d",
        "findings_summary": f"Overall bias score: {report.overall_score:.1f}/100 — {report.overall_verdict.value.upper()}",
        "limitations": "Single-variable counterfactual testing; intersectional bias requires separate audit",
        "remediation_actions": "To be completed based on remediation workshop recommendations",
        "monitoring_plan": "Schedule quarterly re-audits; alert on score increase >10 points",
        "human_oversight": "All fail/concern results require human review before deployment",
        "contact_for_concerns": "To be completed by deploying organisation",
    }


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1:
        raise ValueError("No JSON in response")
    import json
    return json.loads(raw[start:end])
