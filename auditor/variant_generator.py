"""
Variant Generator — pure function, no side-effects.

Takes a prompt template with {{placeholder}} syntax and a demographic matrix,
returns the Cartesian product of all variant (prompt, context) pairs.
"""
from __future__ import annotations
import itertools
import re
from auditor.report_models import VariantPrompt

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def extract_placeholders(template: str) -> list[str]:
    """Return ordered unique placeholders found in template."""
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(template)))


def generate_variants(
    template: str,
    matrix: dict[str, list[str]],
) -> list[VariantPrompt]:
    """
    Generate all variant prompts via Cartesian product.

    Args:
        template: Prompt string with {{variable}} placeholders.
        matrix:   {dimension_name: [value1, value2, ...]}

    Returns:
        List of VariantPrompt — one per combination.

    Raises:
        ValueError: If template contains a placeholder not in matrix.
    """
    placeholders = extract_placeholders(template)
    if not placeholders:
        return [VariantPrompt(rendered=template, context={})]

    # Validate all placeholders have matrix entries
    missing = [p for p in placeholders if p not in matrix]
    if missing:
        raise ValueError(f"Placeholders not in matrix: {missing}")

    # Build ordered list of (dimension, values) matching placeholder order
    dimensions = placeholders
    value_lists = [matrix[d] for d in dimensions]

    variants: list[VariantPrompt] = []
    for combo in itertools.product(*value_lists):
        context = dict(zip(dimensions, combo))
        rendered = template
        for dim, val in context.items():
            rendered = rendered.replace(f"{{{{{dim}}}}}", val)
        variants.append(VariantPrompt(rendered=rendered, context=context))

    return variants


def group_by_dimension(
    variants: list[VariantPrompt],
) -> dict[str, dict[str, list[VariantPrompt]]]:
    """
    Group variants by each demographic dimension and its values.

    Returns: {dimension: {value: [variants]}}
    """
    if not variants or not variants[0].context:
        return {}

    dimensions = list(variants[0].context.keys())
    result: dict[str, dict[str, list[VariantPrompt]]] = {}

    for dim in dimensions:
        result[dim] = {}
        for v in variants:
            val = v.context[dim]
            result[dim].setdefault(val, []).append(v)

    return result
