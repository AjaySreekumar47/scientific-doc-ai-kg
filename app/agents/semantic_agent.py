from __future__ import annotations

import re
from typing import List

from app.pipelines.extract.schema import ExtractionDocument


METRIC_PATTERNS = [
    r"\baccuracy\b",
    r"\bf1\b",
    r"\bprecision\b",
    r"\brecall\b",
    r"\bauc\b",
    r"\bbleu\b",
    r"\brouge\b",
    r"\bdice\b",
    r"\biou\b",
]

SECTION_HINTS = {
    "method": ["method", "methods", "approach", "model", "architecture"],
    "dataset": ["dataset", "data", "benchmark", "corpus"],
    "result": ["result", "results", "performance", "evaluation"],
    "limitation": ["limitation", "limitations", "weakness", "failure"],
    "future_work": ["future work", "next steps", "extensions"],
}


def detect_metrics(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for pattern in METRIC_PATTERNS:
        if re.search(pattern, lower):
            found.append(pattern.replace(r"\b", ""))
    return found


def classify_claim_type(text: str) -> str:
    lower = text.lower()

    for label, keywords in SECTION_HINTS.items():
        for kw in keywords:
            if kw in lower:
                return label

    return "general_claim"


def run_semantic_extraction_agent(doc: ExtractionDocument) -> dict:
    """
    Semantic Extraction Agent:
    derives lightweight scientific claims from extracted blocks.
    """
    claims = []

    for b in doc.blocks:
        if not b.text:
            continue

        text = b.text.strip()
        if len(text) < 40:
            continue

        claim_type = classify_claim_type(text)
        metrics = detect_metrics(text)

        claims.append(
            {
                "source_block_id": b.block_id,
                "page_index": b.page_index,
                "claim_type": claim_type,
                "text": text[:500],
                "metrics_detected": metrics,
            }
        )

    return {
        "agent_name": "semantic_extraction_agent",
        "num_claims": len(claims),
        "claims": claims,
    }