"""Pydantic data models for InjectGuard inspection and decisions."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class InspectionDecision(BaseModel):
    """The structured decision returned by the LLM cumulative judge.
    
    Contains exactly the 3 requested fields:
    - score: cumulative risk/injection score (0.0 to 1.0)
    - is_accepted: boolean indicating whether input is safe (True = accepted, False = rejected)
    - reason: explanation of the decision
    """
    score: float = Field(
        ...,
        description="Cumulative injection risk score between 0.0 (benign) and 1.0 (dangerous injection)",
        ge=0.0,
        le=1.0,
    )
    is_accepted: bool = Field(
        ...,
        description="True if the input is safe to pass to the target LLM; False if rejected/blocked",
    )
    reason: str = Field(
        ...,
        description="Clear reason explaining why the input was accepted or rejected",
    )


class ExtractedContent(BaseModel):
    """Normalized content extracted from text, audio, or image input."""
    modality: Literal["text", "audio", "image"]
    raw_input: Any
    extracted_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
