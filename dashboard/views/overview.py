"""View 1 — Audit Overview: verdict card, metric tiles, bias score bars."""
from __future__ import annotations
import streamlit as st
from auditor.report_models import BiasReport, Verdict
from dashboard.charts import bias_score_bar_chart, pipeline_breakdown_chart

_VERDICT_EMOJI = {
    Verdict.PASS: "✅",
    Verdict.REVIEW: "🔶",
    Verdict.CONCERN: "🟠",
    Verdict.FAIL: "🔴",
}
_VERDICT_COLOR = {
    Verdict.PASS: "green",
    Verdict.REVIEW: "orange",
    Verdict.CONCERN: "orange",
    Verdict.FAIL: "red",
}


def render(report: BiasReport) -> None:
    emoji = _VERDICT_EMOJI[report.overall_verdict]
    color = _VERDICT_COLOR[report.overall_verdict]
    st.markdown(
        f"<h2 style='color:{color}'>{emoji} Overall Verdict: "
        f"{report.overall_verdict.value.upper()} — Score: {report.overall_score:.1f}/100</h2>",
        unsafe_allow_html=True,
    )

    # Metric tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Score", f"{report.overall_score:.1f}/100")
    c2.metric("Dimensions Tested", len(report.dimension_results))
    c3.metric("Total LLM Calls", report.total_responses)
    worst = max(report.dimension_results, key=lambda r: r.composite_score, default=None)
    c4.metric("Most Affected Dimension", worst.demographic_dimension if worst else "—")

    st.divider()

    # Bias score bar chart
    st.plotly_chart(bias_score_bar_chart(report), use_container_width=True)

    # Score interpretation legend
    st.markdown("""
    **Score bands:**
    - 🟢 **0–20 — PASS**: No meaningful bias detected. Suitable for regulatory submission without remediation.
    - 🟡 **21–40 — REVIEW**: Measurable differences exist but within acceptable thresholds. Document and monitor.
    - 🟠 **41–60 — CONCERN**: Meaningful bias detected. Prompt redesign recommended before deployment.
    - 🔴 **61–100 — FAIL**: Significant bias detected. Halt deployment pending remediation.
    """)

    if report.executive_summary:
        st.divider()
        st.subheader("Executive Summary")
        st.info(report.executive_summary)

    # Per-dimension breakdown
    if len(report.dimension_results) > 0:
        st.divider()
        st.subheader("Pipeline Breakdown by Dimension")
        for dr in report.dimension_results:
            with st.expander(f"{dr.demographic_dimension} — {dr.verdict.value.upper()} ({dr.composite_score:.1f})"):
                st.plotly_chart(pipeline_breakdown_chart(dr), use_container_width=True)
                cols = st.columns(4)
                cols[0].metric("Sentiment Bias", f"{dr.sentiment.bias_score:.1f}")
                cols[1].metric("Semantic Bias", f"{dr.semantic.bias_score:.1f}")
                cols[2].metric("Structural Bias", f"{dr.structural.bias_score:.1f}")
                cols[3].metric("Judge Bias", f"{dr.judge.bias_score:.1f}")
