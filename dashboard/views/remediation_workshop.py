"""View 5 — Remediation Workshop: shown only for concern/fail audits."""
from __future__ import annotations
import streamlit as st
from auditor.report_models import BiasReport, Verdict


def render(report: BiasReport) -> None:
    if report.overall_verdict not in (Verdict.CONCERN, Verdict.FAIL):
        st.success(
            f"Verdict is **{report.overall_verdict.value.upper()}** — no remediation required. "
            "Remediation Workshop is shown only for Concern or Fail results."
        )
        return

    st.subheader("Remediation Workshop")
    st.markdown(
        "The system has analysed the bias patterns and generated specific prompt modifications "
        "that should reduce the detected biases. Each recommendation shows the expected impact "
        "and confidence level."
    )

    if not report.remediation:
        st.warning(
            "Remediation analysis not available. Re-run the audit with `enable_judge=True` "
            "and an active Anthropic API key to generate remediation recommendations."
        )
        return

    st.info(f"**Root cause hypothesis:** {report.remediation.root_cause_hypothesis}")
    st.info(f"**Estimated bias reduction:** {report.remediation.estimated_bias_reduction}")

    st.divider()
    for i, rec in enumerate(report.remediation.recommendations, 1):
        confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(rec.confidence, "⚪")
        with st.expander(
            f"{confidence_color} Recommendation {i} — {rec.change_type.replace('_', ' ').title()} [{rec.confidence} confidence]",
            expanded=i == 1,
        ):
            if rec.current_text:
                st.markdown("**Current text:**")
                st.code(rec.current_text, language=None)
            st.markdown("**Suggested replacement:**")
            st.code(rec.suggested_text, language=None)
            st.markdown(f"**Expected impact:** {rec.expected_impact}")

            c1, c2, c3 = st.columns(3)
            if c1.button("Accept", key=f"accept_{i}"):
                st.success("Recommendation accepted — apply to your prompt template.")
            if c2.button("Reject", key=f"reject_{i}"):
                st.info("Recommendation rejected.")
            if c3.button("Flag for review", key=f"flag_{i}"):
                st.warning("Flagged for human review.")

    if report.regulatory_docs:
        st.divider()
        st.subheader("EU AI Act Article 13 Documentation Preview")
        for key, val in report.regulatory_docs.items():
            with st.expander(key.replace("_", " ").title()):
                st.write(val)
