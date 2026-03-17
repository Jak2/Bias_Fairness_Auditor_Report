"""View 4 — Statistical Results Table with CSV export."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from auditor.report_models import BiasReport
from dashboard.charts import structural_quality_radar, semantic_similarity_heatmap


def render(report: BiasReport) -> None:
    st.subheader("Statistical Results — Full Evidence Table")
    st.markdown(
        "Complete ANOVA and effect size results. "
        "Cohen's d: 🔴 ≥0.8 large, 🟠 ≥0.5 medium, 🟢 <0.5 small. "
        "Export to CSV for regulatory documentation."
    )

    rows = []
    for dr in report.dimension_results:
        rows.append({
            "Dimension": dr.demographic_dimension,
            "Pipeline": "Sentiment",
            "F-stat": dr.sentiment.f_statistic,
            "p-value": dr.sentiment.p_value,
            "Cohen's d": dr.sentiment.cohens_d,
            "Significant": dr.sentiment.is_significant,
            "Bias Score": dr.sentiment.bias_score,
        })
        rows.append({
            "Dimension": dr.demographic_dimension,
            "Pipeline": "Structural Quality",
            "F-stat": dr.structural.f_statistic,
            "p-value": dr.structural.p_value,
            "Cohen's d": dr.structural.cohens_d,
            "Significant": dr.structural.is_significant,
            "Bias Score": dr.structural.bias_score,
        })

    df = pd.DataFrame(rows)

    def color_cohens_d(val):
        if val >= 0.8:
            return "background-color: #fee2e2"
        elif val >= 0.5:
            return "background-color: #fef3c7"
        return "background-color: #dcfce7"

    st.dataframe(
        df.style.applymap(color_cohens_d, subset=["Cohen's d"]),
        use_container_width=True,
        hide_index=True,
    )

    csv = df.to_csv(index=False)
    st.download_button("Download CSV for regulatory submission", csv, "bias_audit_statistics.csv", "text/csv")

    st.divider()
    st.subheader("Visual Analysis by Dimension")
    for dr in report.dimension_results:
        with st.expander(f"{dr.demographic_dimension} — Structural Quality & Semantic Similarity"):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(structural_quality_radar(dr), use_container_width=True)
            with c2:
                st.plotly_chart(semantic_similarity_heatmap(dr), use_container_width=True)

            # Structural quality breakdown table
            sq_rows = []
            for grp, sq in dr.structural.results_by_group.items():
                sq_rows.append({
                    "Group": grp,
                    "Avg Words": sq.word_count_mean,
                    "Avg Sentences": sq.sentence_count_mean,
                    "Specificity": f"{sq.specificity_mean:.3f}",
                    "Completeness": f"{sq.completeness_mean:.3f}",
                    "Vocab Complexity": f"{sq.vocabulary_complexity_mean:.2f}",
                })
            st.dataframe(pd.DataFrame(sq_rows), use_container_width=True, hide_index=True)
