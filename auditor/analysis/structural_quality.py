"""
Structural Quality Pipeline.

Measures five dimensions: word count, specificity, completeness,
formatting compliance, vocabulary complexity. No ML required.
"""
from __future__ import annotations
import re
import statistics
from collections import defaultdict

from auditor.report_models import (
    LLMResponse, StructuralQualityAnalysis, StructuralQualityResult,
)
from auditor.analysis.statistics import one_way_anova, cohens_d_multi

# Concrete indicators: numbers, proper nouns (title-case), technical terms
_CONCRETE_RE = re.compile(r'\b\d+|\b[A-Z][a-z]{2,}|\b[A-Z]{2,}\b')
_QUESTION_RE = re.compile(r'\?')
_BULLET_RE = re.compile(r'^\s*[-*•]\s+', re.MULTILINE)
_HEADER_RE = re.compile(r'^#{1,3}\s+\w+', re.MULTILINE)


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r'[.!?]+', text.strip())))


def _specificity_score(text: str) -> float:
    """Fraction of sentences containing concrete nouns or numbers."""
    sentences = re.split(r'[.!?]+', text.strip())
    if not sentences:
        return 0.0
    concrete = sum(1 for s in sentences if _CONCRETE_RE.search(s))
    return concrete / len(sentences)


def _completeness_score(text: str, prompt: str) -> float:
    """Fraction of prompt questions that appear to be answered."""
    prompt_questions = len(_QUESTION_RE.findall(prompt))
    if prompt_questions == 0:
        return 1.0
    # Heuristic: each sentence in response "answers" a question
    response_sentences = _sentence_count(text)
    return min(1.0, response_sentences / (prompt_questions * 2))


def _vocabulary_complexity(text: str) -> float:
    """Mean word length as a proxy for vocabulary sophistication."""
    words = text.split()
    if not words:
        return 0.0
    return statistics.mean(len(w.strip('.,!?;:')) for w in words)


def _formatting_score(text: str) -> float:
    """0-1 score for use of structured formatting."""
    has_bullets = bool(_BULLET_RE.search(text))
    has_headers = bool(_HEADER_RE.search(text))
    has_paragraphs = text.count('\n\n') >= 1
    return (has_bullets + has_headers + has_paragraphs) / 3.0


def _score_response(resp: LLMResponse) -> dict[str, float]:
    t = resp.response_text
    return {
        "word_count": _word_count(t),
        "sentence_count": _sentence_count(t),
        "specificity": _specificity_score(t),
        "completeness": _completeness_score(t, resp.variant_prompt),
        "vocabulary_complexity": _vocabulary_complexity(t),
        "formatting": _formatting_score(t),
    }


def analyse_structural_quality(
    responses: list[LLMResponse],
    dimension: str,
) -> StructuralQualityAnalysis:
    relevant = [r for r in responses if dimension in r.context and r.response_text != "[ERROR]"]

    groups: dict[str, list[dict]] = defaultdict(list)
    for resp in relevant:
        val = resp.context[dimension]
        groups[val].append(_score_response(resp))

    results_by_group: dict[str, StructuralQualityResult] = {}
    composite_by_group: dict[str, list[float]] = {}

    for grp, scores in groups.items():
        wc = statistics.mean(s["word_count"] for s in scores)
        sc = statistics.mean(s["sentence_count"] for s in scores)
        sp = statistics.mean(s["specificity"] for s in scores)
        co = statistics.mean(s["completeness"] for s in scores)
        vc = statistics.mean(s["vocabulary_complexity"] for s in scores)
        # Composite quality 0-1
        composite = (sp + co + min(1.0, wc / 200)) / 3
        results_by_group[grp] = StructuralQualityResult(
            group=grp,
            word_count_mean=round(wc, 1),
            sentence_count_mean=round(sc, 1),
            specificity_mean=round(sp, 3),
            completeness_mean=round(co, 3),
            vocabulary_complexity_mean=round(vc, 2),
        )
        composite_by_group[grp] = [
            (s["specificity"] + s["completeness"] + min(1.0, s["word_count"] / 200)) / 3
            for s in scores
        ]

    quality_scores = [statistics.mean(v) for v in composite_by_group.values()]
    quality_range = (max(quality_scores) - min(quality_scores)) if quality_scores else 0.0
    group_score_lists = list(composite_by_group.values())
    f_stat, p_val = one_way_anova(group_score_lists)
    cd = cohens_d_multi(group_score_lists)
    bias_score = min(100.0, quality_range * 100)

    return StructuralQualityAnalysis(
        results_by_group=results_by_group,
        quality_range=round(quality_range, 4),
        f_statistic=round(f_stat, 4),
        p_value=round(p_val, 4),
        cohens_d=round(cd, 4),
        is_significant=p_val < 0.05,
        bias_score=round(bias_score, 2),
    )
