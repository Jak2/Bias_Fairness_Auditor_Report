"""Shared Plotly chart builders used across all dashboard views."""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from auditor.report_models import BiasReport, DimensionBiasResult, Verdict

_VERDICT_COLOR = {
    Verdict.PASS: "#22c55e",
    Verdict.REVIEW: "#f59e0b",
    Verdict.CONCERN: "#f97316",
    Verdict.FAIL: "#ef4444",
}


def bias_score_bar_chart(report: BiasReport) -> go.Figure:
    """Horizontal bar chart — one bar per dimension, color-coded by verdict."""
    dims = [r.demographic_dimension for r in report.dimension_results]
    scores = [r.composite_score for r in report.dimension_results]
    colors = [_VERDICT_COLOR[r.verdict] for r in report.dimension_results]

    fig = go.Figure(go.Bar(
        x=scores, y=dims, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
    ))
    fig.add_vline(x=20, line_dash="dot", line_color="#22c55e", annotation_text="Pass threshold")
    fig.add_vline(x=60, line_dash="dot", line_color="#ef4444", annotation_text="Fail threshold")
    fig.update_layout(
        title="Composite Bias Score by Demographic Dimension",
        xaxis_title="Bias Score (0=no bias, 100=extreme bias)",
        xaxis=dict(range=[0, 105]),
        height=max(200, len(dims) * 60 + 100),
        margin=dict(l=10, r=40, t=50, b=30),
        plot_bgcolor="#f9fafb",
    )
    return fig


def sentiment_violin_chart(result: DimensionBiasResult) -> go.Figure:
    """Violin plot of sentiment distributions per demographic group."""
    fig = go.Figure()
    for grp, sr in result.sentiment.results_by_group.items():
        fig.add_trace(go.Violin(
            y=sr.scores, name=grp,
            box_visible=True, meanline_visible=True,
            points="all",
        ))
    sig_text = "★ Statistically significant" if result.sentiment.is_significant else "Not significant"
    fig.update_layout(
        title=f"Sentiment Score Distribution — {result.demographic_dimension}<br>"
              f"<sup>ANOVA p={result.sentiment.p_value:.4f} | Cohen's d={result.sentiment.cohens_d:.3f} | {sig_text}</sup>",
        yaxis_title="VADER Compound Score (-1 to +1)",
        yaxis=dict(range=[-1.1, 1.1]),
        plot_bgcolor="#f9fafb",
        height=400,
    )
    return fig


def structural_quality_radar(result: DimensionBiasResult) -> go.Figure:
    """Radar chart comparing structural quality metrics across groups."""
    categories = ["Word Count\n(norm)", "Specificity", "Completeness", "Vocab Complexity\n(norm)"]
    fig = go.Figure()
    for grp, sq in result.structural.results_by_group.items():
        values = [
            min(1.0, sq.word_count_mean / 300),
            sq.specificity_mean,
            sq.completeness_mean,
            min(1.0, sq.vocabulary_complexity_mean / 8),
        ]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=grp,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Structural Quality Comparison — {result.demographic_dimension}",
        height=380,
    )
    return fig


def semantic_similarity_heatmap(result: DimensionBiasResult) -> go.Figure:
    """Simple bar chart showing within vs between group similarity."""
    fig = go.Figure(go.Bar(
        x=["Within-Group Similarity", "Between-Group Similarity"],
        y=[result.semantic.within_group_mean, result.semantic.between_group_mean],
        marker_color=["#3b82f6", "#ef4444" if result.semantic.similarity_gap > 0.1 else "#22c55e"],
        text=[f"{result.semantic.within_group_mean:.3f}", f"{result.semantic.between_group_mean:.3f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Semantic Similarity — {result.demographic_dimension}<br>"
              f"<sup>Gap={result.semantic.similarity_gap:.3f} | Bias Score={result.semantic.bias_score:.1f}</sup>",
        yaxis=dict(range=[0, 1.1]),
        yaxis_title="Cosine Similarity",
        plot_bgcolor="#f9fafb",
        height=350,
    )
    return fig


def pipeline_breakdown_chart(result: DimensionBiasResult) -> go.Figure:
    """Stacked bar showing contribution of each pipeline to composite score."""
    from config import get_settings
    cfg = get_settings()
    w = cfg.bias_score_weights
    contributions = {
        "Sentiment": result.sentiment.bias_score * w["sentiment"],
        "Semantic": result.semantic.bias_score * w["semantic"],
        "Structural": result.structural.bias_score * w["structural"],
        "Judge": result.judge.bias_score * w["judge"],
    }
    colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]
    fig = go.Figure()
    for (name, val), color in zip(contributions.items(), colors):
        fig.add_trace(go.Bar(name=name, x=[result.demographic_dimension], y=[val], marker_color=color))
    fig.update_layout(
        barmode="stack",
        title=f"Pipeline Contribution to Composite Score — {result.demographic_dimension}",
        yaxis_title="Weighted Contribution",
        yaxis=dict(range=[0, 105]),
        height=320,
    )
    return fig


def historical_trend_chart(jobs: list[dict]) -> go.Figure:
    """Line chart of overall bias score over time for trend monitoring."""
    if not jobs:
        return go.Figure()
    df = pd.DataFrame(jobs)
    df = df.dropna(subset=["score", "created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at")
    fig = px.line(
        df, x="created_at", y="score", color="matrix",
        markers=True,
        title="Bias Score Trend Over Time",
        labels={"score": "Composite Bias Score", "created_at": "Date", "matrix": "Matrix"},
    )
    fig.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.05, annotation_text="Pass zone")
    fig.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.05, annotation_text="Fail zone")
    fig.update_layout(height=350, plot_bgcolor="#f9fafb")
    return fig
