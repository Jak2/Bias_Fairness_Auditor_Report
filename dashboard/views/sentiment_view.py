"""View 2 — Sentiment Analysis: violin plots + significance table."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from auditor.report_models import BiasReport
from dashboard.charts import sentiment_violin_chart


def render(report: BiasReport) -> None:
    st.subheader("Sentiment Analysis — Tonal Bias Detection")
    st.markdown(
        "Violin plots show the full distribution of VADER sentiment scores per demographic group. "
        "A shift in position or shape indicates **tonal bias** — the LLM is measurably warmer or "
        "colder toward some groups. Stars (★) mark statistically significant findings (p < 0.05)."
    )

    for dr in report.dimension_results:
        st.divider()
        st.markdown(f"#### Dimension: `{dr.demographic_dimension}`")
        st.plotly_chart(sentiment_violin_chart(dr), use_container_width=True)

        # Stats table
        rows = []
        for grp, sr in dr.sentiment.results_by_group.items():
            rows.append({
                "Group": grp,
                "Mean Sentiment": f"{sr.mean:+.4f}",
                "Std Dev": f"{sr.std:.4f}",
                "N Samples": len(sr.scores),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        sig_badge = "🔴 **Significant**" if dr.sentiment.is_significant else "🟢 Not significant"
        st.markdown(
            f"ANOVA: F={dr.sentiment.f_statistic:.3f}, p={dr.sentiment.p_value:.4f} — {sig_badge}  \n"
            f"Cohen's d (worst pairwise): **{dr.sentiment.cohens_d:.3f}** "
            f"({'Large ≥0.8' if dr.sentiment.cohens_d >= 0.8 else 'Medium ≥0.5' if dr.sentiment.cohens_d >= 0.5 else 'Small <0.5'} effect)"
        )
