from __future__ import annotations

from app.pipelines.extract.schema import ExtractionDocument


def run_layout_block_agent(doc: ExtractionDocument) -> dict:
    """
    Layout & Block Agent:
    filters block-level structural information from the extracted document.
    """
    layout_blocks = [
        {
            "block_id": b.block_id,
            "type": b.type.value,
            "page_index": b.page_index,
            "bbox_norm": b.bbox_norm.model_dump(),
            "text": b.text,
        }
        for b in doc.blocks
    ]

    return {
        "agent_name": "layout_block_agent",
        "num_blocks": len(layout_blocks),
        "blocks": layout_blocks,
    }