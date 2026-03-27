from __future__ import annotations

from app.pipelines.extract.schema import ExtractionDocument


def run_figure_table_agent(doc: ExtractionDocument) -> dict:
    """
    Figure/Table Agent:
    extracts figure and table metadata from the document.
    """
    figures = [
        {
            "figure_id": f.figure_id,
            "page_index": f.page_index,
            "bbox_norm": f.bbox_norm.model_dump(),
            "caption_block_id": f.caption_block_id,
            "image_path": f.image_path,
        }
        for f in doc.figures
    ]

    tables = [
        {
            "table_id": t.table_id,
            "page_index": t.page_index,
            "bbox_norm": t.bbox_norm.model_dump(),
            "caption_block_id": t.caption_block_id,
            "image_path": t.image_path,
            "num_cells": len(t.cells),
        }
        for t in doc.tables
    ]

    return {
        "agent_name": "figure_table_agent",
        "num_figures": len(figures),
        "num_tables": len(tables),
        "figures": figures,
        "tables": tables,
    }