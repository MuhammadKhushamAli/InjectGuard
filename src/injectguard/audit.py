"""Audit and Quarantine logging module for InjectGuard.

Implements section 4.2 of the proposal:
"Suspicious chunks are quarantined or wrapped in a clearly marked untrusted-data block,
and every decision is written to an audit log."
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from injectguard.models import InspectionDecision

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_LOG = PROJECT_ROOT / "eval" / "reports" / "audit.jsonl"


class AuditRecord(BaseModel):
    """Structured audit trail record."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str  # "INCORPORATED" or "QUARANTINED"
    modality: str
    decision: InspectionDecision
    raw_content_preview: str
    quarantined: bool


class AuditLogger:
    """Manages appending audit logs and querying quarantine records."""

    def __init__(self, log_path: Path | str = DEFAULT_AUDIT_LOG) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        modality: str,
        decision: InspectionDecision,
        raw_content: Any,
    ) -> AuditRecord:
        """Record an inspection decision to disk."""
        preview = str(raw_content)
        if len(preview) > 200:
            preview = preview[:200] + "... [TRUNCATED]"

        record = AuditRecord(
            event_type=event_type,
            modality=modality,
            decision=decision,
            raw_content_preview=preview,
            quarantined=not decision.is_accepted,
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        return record
