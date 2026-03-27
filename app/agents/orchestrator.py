from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.agents.figure_table_agent import run_figure_table_agent
from app.agents.layout_agent import run_layout_block_agent
from app.agents.planner import build_execution_plan
from app.agents.semantic_agent import run_semantic_extraction_agent
from app.pipelines.extract.run import extract_pymupdf_basic

from app.agents.verification_agent import verify_claims


def run_agentic_extraction(
    pdf_path: Path,
    *,
    original_name: str,
    doc_id: str,
    assets_root: Path,
    user_goal: Optional[str] = None,
) -> dict:
    """
    Orchestrates planner + extraction-oriented agents.
    """
    plan = build_execution_plan(pdf_path, user_goal=user_goal)
    plan.document_name = original_name

    doc = extract_pymupdf_basic(
        pdf_path,
        doc_id=doc_id,
        assets_root=assets_root,
    )
    doc.pdf_filename = original_name

    layout_result = run_layout_block_agent(doc)
    figure_table_result = run_figure_table_agent(doc)
    semantic_result = run_semantic_extraction_agent(doc)

    verification_result = verify_claims(
        claims=semantic_result["claims"],
        blocks=layout_result["blocks"],
        tables=figure_table_result["tables"],
    )

    verified_only = [
    c for c in verification_result["verified_claims"]
    if c["verification_status"] == "verified"
    ]

    return {
        "document_id": doc.document_id,
        "document_name": doc.pdf_filename,
        "execution_plan": plan.model_dump(),
        "agent_outputs": {
            "layout_block_agent": layout_result,
            "figure_table_agent": figure_table_result,
            "semantic_extraction_agent": semantic_result,
            "verification_agent": verification_result,
        },
        "verified_claims": verified_only,
        "num_verified_claims": len(verified_only),
        "raw_annotations": doc.annotations,
    }