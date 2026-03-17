"""View 3 — Response Explorer: side-by-side response comparison."""
from __future__ import annotations
import streamlit as st
from auditor.report_models import BiasReport, LLMResponse


def render(report: BiasReport, all_responses: list[LLMResponse]) -> None:
    st.subheader("Response Explorer — Side-by-Side Comparison")
    st.markdown(
        "Compare actual LLM responses across demographic groups. "
        "Select a demographic dimension and two groups to compare. "
        "The **Most Divergent Pair** panel highlights the worst-case content difference."
    )

    if not all_responses:
        st.warning("No responses available for this audit.")
        return

    # Dimension selector
    dims = list({r.context_key: True for dr in report.dimension_results for r in []}.keys())
    dims = [dr.demographic_dimension for dr in report.dimension_results]
    if not dims:
        st.info("No demographic dimensions found.")
        return

    sel_dim = st.selectbox("Select dimension", dims)
    dr = next((r for r in report.dimension_results if r.demographic_dimension == sel_dim), None)

    # Get responses for this dimension
    dim_responses: dict[str, list[LLMResponse]] = {}
    for resp in all_responses:
        if sel_dim in resp.context:
            val = resp.context[sel_dim]
            dim_responses.setdefault(val, []).append(resp)

    groups = list(dim_responses.keys())
    if len(groups) < 2:
        st.info("Need at least 2 groups to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        grp_a = st.selectbox("Group A", groups, index=0, key="grp_a")
    with col2:
        grp_b = st.selectbox("Group B", groups, index=min(1, len(groups)-1), key="grp_b")

    st.divider()

    # Side-by-side representative responses (run 1)
    resp_a = next((r for r in dim_responses.get(grp_a, []) if r.run_number == 1), None)
    resp_b = next((r for r in dim_responses.get(grp_b, []) if r.run_number == 1), None)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Group A: {grp_a}**")
        if resp_a:
            st.text_area("Response", resp_a.response_text, height=300, key="ra", disabled=True)
            st.caption(f"Latency: {resp_a.latency_ms:.0f}ms | Tokens: {resp_a.token_count}")
        else:
            st.info("No response found")
    with c2:
        st.markdown(f"**Group B: {grp_b}**")
        if resp_b:
            st.text_area("Response", resp_b.response_text, height=300, key="rb", disabled=True)
            st.caption(f"Latency: {resp_b.latency_ms:.0f}ms | Tokens: {resp_b.token_count}")
        else:
            st.info("No response found")

    # Most divergent pair highlight
    if dr and dr.semantic.most_divergent_pair:
        st.divider()
        st.markdown("#### Most Divergent Pair (lowest semantic similarity)")
        ga, gb = dr.semantic.most_divergent_pair
        st.info(
            f"Groups **{ga}** vs **{gb}** — cosine similarity: **{dr.semantic.most_divergent_score:.3f}** "
            f"(1.0 = identical content, 0.0 = completely different)"
        )

    # Judge assessment results
    if dr and dr.judge.comparisons:
        st.divider()
        st.subheader("LLM Judge Assessments (Blind)")
        for comp in dr.judge.comparisons[:3]:
            with st.expander(f"Comparison — Overall signal: {comp.overall_bias_signal.value.upper()}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Quality A", f"{comp.response_a_quality:.2f}")
                c2.metric("Quality B", f"{comp.response_b_quality:.2f}")
                c3.metric("Equivalent?", "Yes" if comp.quality_equivalent else "No")
                if comp.tone_description:
                    st.markdown(f"**Tone:** {comp.tone_description}")
                if comp.substance_description:
                    st.markdown(f"**Substance:** {comp.substance_description}")
                if comp.primary_concern:
                    st.warning(f"Primary concern: {comp.primary_concern}")
