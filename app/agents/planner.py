from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz

from app.agents.models import AgentDecision, ExecutionPlan


def estimate_scanned_pdf(pdf_path: Path, text_len_threshold: int = 20) -> bool:
    """
    Simple heuristic:
    if most pages have almost no native text, treat document as scanned.
    """
    with fitz.open(pdf_path) as pdf:
        if pdf.page_count == 0:
            return False

        scanned_like_pages = 0

        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            text = page.get_text("text").strip()
            if len(text) < text_len_threshold:
                scanned_like_pages += 1

        return scanned_like_pages >= max(1, pdf.page_count // 2)


def build_execution_plan(
    pdf_path: Path,
    *,
    user_goal: Optional[str] = None,
) -> ExecutionPlan:
    """
    Planner Agent:
    decides which downstream agents should run.
    """
    is_scanned = estimate_scanned_pdf(pdf_path)
    goals = [user_goal] if user_goal else ["build scientific knowledge graph"]

    agent_sequence = [
        "planner_agent",
        "layout_block_agent",
        "figure_table_agent",
        "semantic_extraction_agent",
        "verification_agent",
        "rdf_graph_builder_agent",
    ]

    decisions = [
        AgentDecision(
            agent_name="layout_block_agent",
            should_run=True,
            reason="Core block extraction is required for all documents.",
        ),
        AgentDecision(
            agent_name="figure_table_agent",
            should_run=True,
            reason="Figures and tables are required for provenance-grounded scientific extraction.",
        ),
        AgentDecision(
            agent_name="semantic_extraction_agent",
            should_run=True,
            reason="Scientific entities and claims must be derived from extracted content.",
        ),
        AgentDecision(
            agent_name="verification_agent",
            should_run=True,
            reason="Claims should be validated against source evidence before graph insertion.",
        ),
        AgentDecision(
            agent_name="rdf_graph_builder_agent",
            should_run=True,
            reason="Validated outputs must be committed into the knowledge graph.",
        ),
    ]

    if is_scanned:
        notes = "Document appears scanned; OCR should be enabled."
    else:
        notes = "Document appears digitally readable; native extraction should be preferred."

    return ExecutionPlan(
        document_name=pdf_path.name,
        is_probably_scanned=is_scanned,
        run_ocr=is_scanned,
        mode="single_document",
        goals=goals,
        agent_sequence=agent_sequence,
        decisions=decisions,
        notes=notes,
    )