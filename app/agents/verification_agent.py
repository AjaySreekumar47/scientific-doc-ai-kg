from __future__ import annotations

from typing import List, Dict


def compute_overlap_score(claim_text: str, source_text: str) -> float:
    """
    Simple lexical overlap score.
    """
    claim_tokens = set(claim_text.lower().split())
    source_tokens = set(source_text.lower().split())

    if not claim_tokens:
        return 0.0

    overlap = claim_tokens.intersection(source_tokens)
    return len(overlap) / len(claim_tokens)


def verify_claims(
    claims: List[Dict],
    blocks: List[Dict],
    tables: List[Dict],
) -> dict:
    """
    Verification Agent:
    checks if claims are grounded in extracted evidence.
    """

    verified_claims = []

    for claim in claims:
        claim_text = claim["text"]

        best_score = 0.0
        best_block_ids = []

        # Check against text blocks
        for b in blocks:
            score = compute_overlap_score(claim_text, b.get("text", ""))

            if score > best_score:
                best_score = score
                best_block_ids = [b["block_id"]]

        # Optional: boost if claim mentions table-related metrics
        for t in tables:
            if any(metric in claim_text.lower() for metric in ["accuracy", "f1", "precision", "recall"]):
                best_score += 0.05

        # Classification
        if best_score > 0.6:
            status = "verified"
        elif best_score > 0.3:
            status = "weak"
        else:
            status = "rejected"

        verified_claims.append(
            {
                **claim,
                "verification_status": status,
                "confidence": round(best_score, 3),
                "evidence_blocks": best_block_ids,
            }
        )

    return {
        "agent_name": "verification_agent",
        "num_claims": len(claims),
        "verified_claims": verified_claims,
    }