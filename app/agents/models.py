from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    agent_name: str
    should_run: bool
    reason: str


class ExecutionPlan(BaseModel):
    document_name: str
    document_type: str = "pdf"
    is_probably_scanned: bool
    run_ocr: bool
    mode: str = "single_document"
    goals: List[str] = Field(default_factory=list)
    agent_sequence: List[str] = Field(default_factory=list)
    decisions: List[AgentDecision] = Field(default_factory=list)
    notes: Optional[str] = None