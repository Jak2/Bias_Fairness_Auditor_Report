"""
Statistical testing utilities.

All pure functions — no side-effects, easily testable.
"""
from __future__ import annotations
import math
import statistics
from scipy import stats  # type: ignore


def one_way_anova(groups: list[list[float]]) -> tuple[float, float]:
    """
    One-way ANOVA across groups.
    Returns (F-statistic, p-value).
    Falls back to (0.0, 1.0) if not enough data.
    """
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return 0.0, 1.0
    try:
        f, p = stats.f_oneway(*valid)
        return float(f) if not math.isnan(f) else 0.0, float(p) if not math.isnan(p) else 1.0
    except Exception:
        return 0.0, 1.0


def chi_squared_test(observed: list[list[int]]) -> tuple[float, float]:
    """
    Chi-squared test for independence on a contingency table.
    Returns (chi2-statistic, p-value).
    """
    try:
        chi2, p, _, _ = stats.chi2_contingency(observed)
        return float(chi2), float(p)
    except Exception:
        return 0.0, 1.0


def cohens_d_two(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two groups."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_diff = abs(statistics.mean(a) - statistics.mean(b))
    pooled_std = math.sqrt(
        ((len(a) - 1) * statistics.variance(a) + (len(b) - 1) * statistics.variance(b))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return round(mean_diff / pooled_std, 4)


def cohens_d_multi(groups: list[list[float]]) -> float:
    """
    Maximum Cohen's d across all pairwise group combinations.
    Represents the worst-case pairwise bias.
    """
    if len(groups) < 2:
        return 0.0
    max_d = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            d = cohens_d_two(groups[i], groups[j])
            max_d = max(max_d, d)
    return max_d


def coefficient_of_variation(values: list[float]) -> float:
    """CV = (std / mean) * 100. Returns 0 if mean is zero."""
    if not values or statistics.mean(values) == 0:
        return 0.0
    return round((statistics.stdev(values) / abs(statistics.mean(values))) * 100, 2)


def two_way_anova_interaction(
    data: list[float],
    factor_a: list[str],
    factor_b: list[str],
) -> tuple[float, float]:
    """
    Simplified two-way ANOVA interaction effect using OLS.
    Returns (F-interaction, p-interaction).
    """
    try:
        import pandas as pd  # type: ignore
        from statsmodels.formula.api import ols  # type: ignore
        from statsmodels.stats.anova import anova_lm  # type: ignore

        df = pd.DataFrame({"y": data, "A": factor_a, "B": factor_b})
        model = ols("y ~ C(A) + C(B) + C(A):C(B)", data=df).fit()
        table = anova_lm(model, typ=2)
        f = float(table.loc["C(A):C(B)", "F"])
        p = float(table.loc["C(A):C(B)", "PR(>F)"])
        return f, p
    except Exception:
        return 0.0, 1.0
