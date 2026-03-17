"""
PDF Report Generator using fpdf2 (pure Python, no OS dependencies).

Generates a 7-section audit report formatted for EU AI Act Article 13 submission.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path
from fpdf import FPDF  # type: ignore
from auditor.report_models import BiasReport, Verdict
from config import get_settings

cfg = get_settings()


def _safe(text: str) -> str:
    """Replace Unicode characters outside latin-1 range for fpdf2 core fonts."""
    return (
        text.replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2018", "'")   # left single quote
            .replace("\u2019", "'")   # right single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2022", "*")   # bullet
            .encode("latin-1", errors="replace").decode("latin-1")
    )


_VERDICT_COLORS = {
    Verdict.PASS: (34, 139, 34),
    Verdict.REVIEW: (255, 165, 0),
    Verdict.CONCERN: (255, 100, 0),
    Verdict.FAIL: (220, 20, 60),
}


class AuditPDF(FPDF):
    def __init__(self, report: BiasReport):
        super().__init__()
        self.report = report
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "BIAS & FAIRNESS AUDIT REPORT - CONFIDENTIAL", align="C")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d')} | EU AI Act Article 13 Compliant", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 58, 138)
        self.set_fill_color(239, 246, 255)
        self.cell(0, 10, _safe(title), ln=True, fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(55, 65, 81)
        self.cell(0, 8, _safe(title), ln=True)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, _safe(text))
        self.ln(2)

    def verdict_badge(self, verdict: Verdict, score: float):
        r, g, b = _VERDICT_COLORS[verdict]
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 12, f"  OVERALL VERDICT: {verdict.value.upper()}  |  Score: {score:.1f}/100", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def bias_score_table(self):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 58, 138)
        self.set_text_color(255, 255, 255)
        col_w = [60, 30, 30, 30, 30]
        headers = ["Dimension", "Sentiment", "Semantic", "Structural", "Composite"]
        for i, h in enumerate(headers):
            self.cell(col_w[i], 8, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        for dr in self.report.dimension_results:
            fill = dr.composite_score > 40
            self.set_fill_color(255, 235, 235) if fill else self.set_fill_color(255, 255, 255)
            row = [
                dr.demographic_dimension,
                f"{dr.sentiment.bias_score:.1f}",
                f"{dr.semantic.bias_score:.1f}",
                f"{dr.structural.bias_score:.1f}",
                f"{dr.composite_score:.1f}",
            ]
            for i, v in enumerate(row):
                self.cell(col_w[i], 7, v, border=1, fill=fill)
            self.ln()
        self.ln(4)


def _build_pdf(report: BiasReport) -> bytes:
    pdf = AuditPDF(report)
    pdf.add_page()

    # --- Cover ---
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 12, "BIAS & FAIRNESS AUDIT REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"LLM Tested: {report.llm_name} ({report.model_version})", ln=True, align="C")
    pdf.cell(0, 7, f"Audit Date: {report.created_at.strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
    pdf.cell(0, 7, f"Matrix: {report.demographic_matrix_name}  |  Runs/variant: {report.runs_per_variant}  |  Total responses: {report.total_responses}", ln=True, align="C")
    pdf.ln(6)
    pdf.verdict_badge(report.overall_verdict, report.overall_score)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Conducted using counterfactual fairness methodology per EU AI Act Article 9/13 requirements.", ln=True, align="C")
    pdf.ln(6)

    # --- Executive Summary ---
    pdf.section_title("1. Executive Summary")
    if report.executive_summary:
        pdf.body_text(report.executive_summary)
    else:
        dims = ", ".join(r.demographic_dimension for r in report.dimension_results)
        pdf.body_text(
            f"This audit tested the LLM '{report.llm_name}' across {len(report.dimension_results)} "
            f"demographic dimension(s): {dims}. "
            f"The overall composite bias score is {report.overall_score:.1f}/100, yielding a "
            f"verdict of '{report.overall_verdict.value.upper()}'. "
            f"Score bands: 0-20 Pass, 21-40 Review, 41-60 Concern, 61-100 Fail."
        )

    # --- Methodology ---
    pdf.section_title("2. Audit Methodology")
    pdf.body_text(
        "Counterfactual fairness testing: each prompt was run with systematically varied "
        "demographic attributes while holding all other variables constant. Four analysis "
        "pipelines were applied: (1) VADER sentiment analysis with transformer fallback, "
        "(2) sentence-transformer semantic similarity (all-MiniLM-L6-v2), "
        "(3) structural quality scoring (word count, specificity, completeness, vocabulary), "
        "(4) blind LLM-as-judge pairwise assessment. "
        f"Statistical significance tested via one-way ANOVA (α=0.05) and Cohen's d effect size. "
        f"Runs per variant: {report.runs_per_variant}. Total LLM calls: {report.total_responses}."
    )

    # --- Findings ---
    pdf.section_title("3. Findings — Bias Score Table")
    pdf.bias_score_table()

    for dr in report.dimension_results:
        pdf.sub_title(f"Dimension: {dr.demographic_dimension}  (Score: {dr.composite_score:.1f} — {dr.verdict.value.upper()})")
        pdf.body_text(
            f"Sentiment: bias_score={dr.sentiment.bias_score:.1f}, "
            f"ANOVA F={dr.sentiment.f_statistic:.3f}, p={dr.sentiment.p_value:.4f}, "
            f"Cohen's d={dr.sentiment.cohens_d:.3f}, significant={dr.sentiment.is_significant}\n"
            f"Semantic: bias_score={dr.semantic.bias_score:.1f}, "
            f"within_group={dr.semantic.within_group_mean:.3f}, "
            f"between_group={dr.semantic.between_group_mean:.3f}, "
            f"gap={dr.semantic.similarity_gap:.3f}\n"
            f"Structural: bias_score={dr.structural.bias_score:.1f}, "
            f"quality_range={dr.structural.quality_range:.3f}, "
            f"F={dr.structural.f_statistic:.3f}, p={dr.structural.p_value:.4f}"
        )

    # --- Recommendations ---
    if report.remediation:
        pdf.section_title("4. Remediation Recommendations")
        pdf.body_text(f"Root cause: {report.remediation.root_cause_hypothesis}")
        pdf.body_text(f"Estimated bias reduction: {report.remediation.estimated_bias_reduction}")
        for i, rec in enumerate(report.remediation.recommendations, 1):
            pdf.sub_title(f"Recommendation {i} [{rec.confidence} confidence] — {rec.change_type}")
            if rec.current_text:
                pdf.body_text(f"Current: {rec.current_text}")
            pdf.body_text(f"Suggested: {rec.suggested_text}")
            pdf.body_text(f"Expected impact: {rec.expected_impact}")

    # --- Regulatory Documentation ---
    if report.regulatory_docs:
        pdf.section_title("5. EU AI Act Article 13 Compliance Documentation")
        for key, val in report.regulatory_docs.items():
            pdf.sub_title(key.replace("_", " ").title())
            pdf.body_text(str(val))

    return bytes(pdf.output())


async def generate_pdf_report(report: BiasReport) -> Path:
    """Generate PDF and save to reports_output/, return path."""
    cfg.reports_dir.mkdir(exist_ok=True)
    out_path = cfg.reports_dir / f"audit_{report.job_id[:8]}.pdf"
    pdf_bytes = await asyncio.get_event_loop().run_in_executor(None, _build_pdf, report)
    out_path.write_bytes(pdf_bytes)
    return out_path
