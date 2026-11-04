"""Rule and Heuristic prompt injection detector for InjectGuard."""

from __future__ import annotations

import re
from typing import Any

# High-risk indirect prompt injection patterns
INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"(?i)\bsystem\s+override\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior)\s+rules\b",
    r"(?i)\breveal\s+(the\s+)?(system\s+prompt|secret|api[_\s-]?key|credentials)\b",
    r"(?i)\byou\s+are\s+now\s+(in\s+developer\s+mode|dan|an\s+unrestricted\s+ai)\b",
    r"(?i)\bexfiltrat(e|ion)\b",
    r"(?i)https?://[^\s]+(?:\.webhook|\.requestcatcher|hook|leak)",
    r"(?i)<\s*system\s*>",
    r"(?i)\[\s*system\s*\]",
]


class RuleDetector:
    """Layer 2 pattern matching detector."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = [re.compile(p) for p in (patterns or INJECTION_PATTERNS)]

    def predict_score(self, text: str) -> float:
        """Return match density/severity score (0.0 to 1.0)."""
        matches = [p.pattern for p in self.patterns if p.search(text)]
        if not matches:
            return 0.0
        # If any high-confidence injection pattern is matched, score high
        return min(1.0, 0.5 + 0.25 * len(matches))

    def predict(self, text: str) -> dict[str, Any]:
        matched_rules = [p.pattern for p in self.patterns if p.search(text)]
        score = self.predict_score(text)
        return {
            "score": score,
            "matched_rules": matched_rules,
            "is_injection": len(matched_rules) > 0,
        }
