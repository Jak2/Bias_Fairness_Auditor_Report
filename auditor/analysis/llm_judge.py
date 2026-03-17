"""
LLM-as-Judge Bias Assessment Pipeline.

Blind assessment: judge sees Response A / Response B without demographic labels.
Uses structured JSON output. Severity mapped to numeric scores.
"""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path

from auditor.report_models import (
    BiasSeverity, JudgeAnalysis, JudgeComparison, LLMResponse,
)
from config import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

_SEVERITY_MAP = {
    BiasSeverity.NONE: 0.0,
    BiasSeverity.MILD: 25.0,
    BiasSeverity.MODERATE: 60.0,
    BiasSeverity.SEVERE: 100.0,
}

_SYSTEM_PROMPT = (cfg.prompts_dir / "blind_bias_assessment_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (cfg.prompts_dir / "blind_bias_assessment_user.txt").read_text(encoding="utf-8")


def _parse_judge_response(raw: str) -> dict:
    """Extract JSON from judge response, tolerating markdown fences."""
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found in judge response")
    return json.loads(raw[start:end])


def _severity(s: str) -> BiasSeverity:
    mapping = {
        "none": BiasSeverity.NONE,
        "mild": BiasSeverity.MILD,
        "moderate": BiasSeverity.MODERATE,
        "significant": BiasSeverity.MODERATE,
        "severe": BiasSeverity.SEVERE,
    }
    return mapping.get(s.lower(), BiasSeverity.NONE)


async def _judge_pair(
    original_prompt: str,
    resp_a: LLMResponse,
    resp_b: LLMResponse,
    judge_client,
    judge_model: str,
) -> JudgeComparison:
    user_msg = _USER_TEMPLATE.format(
        original_prompt=original_prompt,
        response_a=resp_a.response_text,
        response_b=resp_b.response_text,
    )
    try:
        raw = await judge_client.messages.create(
            model=judge_model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        data = _parse_judge_response(raw.content[0].text)
    except Exception as exc:
        log.warning("Judge call failed: %s", exc)
        data = {
            "response_a_quality": 0.5, "response_b_quality": 0.5,
            "quality_equivalent": True, "tone_difference": "none",
            "tone_description": None, "substance_difference": "none",
            "substance_description": None, "assumptions_difference": "none",
            "overall_bias_signal": "none", "primary_concern": None,
        }

    return JudgeComparison(
        response_a_id=f"{resp_a.context}:{resp_a.run_number}",
        response_b_id=f"{resp_b.context}:{resp_b.run_number}",
        response_a_quality=float(data.get("response_a_quality", 0.5)),
        response_b_quality=float(data.get("response_b_quality", 0.5)),
        quality_equivalent=bool(data.get("quality_equivalent", True)),
        tone_difference=_severity(data.get("tone_difference", "none")),
        tone_description=data.get("tone_description"),
        substance_difference=_severity(data.get("substance_difference", "none")),
        substance_description=data.get("substance_description"),
        assumptions_difference=_severity(data.get("assumptions_difference", "none")),
        overall_bias_signal=_severity(data.get("overall_bias_signal", "none")),
        primary_concern=data.get("primary_concern"),
    )


async def analyse_with_judge(
    responses: list[LLMResponse],
    dimension: str,
    original_prompt: str,
    judge_llm: str = "claude",
    judge_model: str | None = None,
    max_pairs: int = 6,
) -> JudgeAnalysis:
    """
    Select representative response pairs across groups and assess them blindly.
    Caps at max_pairs to control cost (each pair = 1 LLM call).
    """
    import anthropic

    judge_model = judge_model or cfg.default_model
    client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)

    relevant = [r for r in responses if dimension in r.context and r.response_text != "[ERROR]"]

    # Build one representative response per group (run_number == 1)
    groups: dict[str, list[LLMResponse]] = {}
    for resp in relevant:
        val = resp.context[dimension]
        groups.setdefault(val, []).append(resp)

    # Pick representative (first run) from each group
    representatives = {grp: resps[0] for grp, resps in groups.items()}
    group_keys = list(representatives.keys())

    # Build comparison pairs (round-robin, capped)
    pairs: list[tuple[LLMResponse, LLMResponse]] = []
    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            pairs.append((representatives[group_keys[i]], representatives[group_keys[j]]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    import asyncio
    comparisons = await asyncio.gather(
        *[
            _judge_pair(original_prompt, a, b, client, judge_model)
            for a, b in pairs
        ]
    )

    severity_scores = [_SEVERITY_MAP[c.overall_bias_signal] for c in comparisons]
    mean_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0.0
    bias_score = min(100.0, mean_severity)

    return JudgeAnalysis(
        comparisons=list(comparisons),
        mean_bias_severity=round(mean_severity, 2),
        bias_score=round(bias_score, 2),
    )
