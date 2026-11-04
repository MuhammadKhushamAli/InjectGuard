"""Layer 4: Egress and Output Verification Guard for InjectGuard.

Implements model output scanning for:
1. Canary token leakage (indirect prompt injection exfiltration).
2. Unexpected/suspicious external URLs, markdown link exploits, and webhooks.
3. System prompt leak patterns.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

# High-risk egress link patterns (attacker webhooks, requestcatchers, suspicious data sinks)
SUSPICIOUS_LINK_PATTERNS = [
    r"https?://[^\s\"')]+(?:\.webhook|\.requestcatcher|hook|leak|exfiltrat|pipedream|ngrok)",
    r"\[[^\]]*\]\((https?://[^\s\"')]+)\)",  # Markdown disguised links
    r"!\[[^\]]*\]\((https?://[^\s\"')]+)\)", # Markdown disguised image exfiltration (get image ping)
]

# Sensitive credentials & prompt leakage indicators
LEAK_SIGNATURES = [
    r"(?i)\bOPENAI_API_KEY\s*[:=]\s*sk-[A-Za-z0-9_-]{20,}",
    r"(?i)\bAWS_SECRET_ACCESS_KEY\s*[:=]\s*[A-Za-z0-9/+=]{40}",
    r"(?i)\bSECRET_KEY\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}['\"]",
]


class EgressVerificationResult(BaseModel):
    """Result of scanning model output before sending to user."""
    is_safe: bool = Field(..., description="True if output is safe to return to user; False if exfiltration detected")
    sanitized_output: str = Field(..., description="Cleaned response, or fallback warning if blocked")
    canary_leaked: bool = Field(False, description="True if the session canary token was leaked in output")
    suspicious_urls_found: list[str] = Field(default_factory=list, description="Any detected exfiltration endpoints")
    sensitive_patterns_matched: list[str] = Field(default_factory=list, description="Any leaked secrets or credentials")
    reasons: list[str] = Field(default_factory=list, description="Descriptions of why egress check failed")


class EgressGuard:
    """Layer 4: Inspects model generated responses before egress."""

    def __init__(
        self,
        block_canary_leak: bool = True,
        block_suspicious_urls: bool = True,
        block_sensitive_patterns: bool = True,
    ) -> None:
        self.block_canary_leak = block_canary_leak
        self.block_suspicious_urls = block_suspicious_urls
        self.block_sensitive_patterns = block_sensitive_patterns

        self._url_regexes = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_LINK_PATTERNS]
        self._leak_regexes = [re.compile(p) for p in LEAK_SIGNATURES]

    def verify_output(
        self,
        llm_output: str,
        canary_token: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> EgressVerificationResult:
        """Scan LLM response for canary token and malicious exfiltration channels.

        Args:
            llm_output: Raw generated text from target LLM.
            canary_token: Active secret canary token for this session.
            allowed_domains: Whitelisted URL domains (if any).

        Returns:
            EgressVerificationResult indicating whether output is safe or blocked.
        """
        reasons: list[str] = []
        canary_leaked = False
        suspicious_urls: list[str] = []
        sensitive_matches: list[str] = []

        # 1. Canary Token Check
        if self.block_canary_leak and canary_token:
            if canary_token.lower() in llm_output.lower():
                canary_leaked = True
                reasons.append(
                    f"CRITICAL: Canary token '{canary_token[:10]}...' was detected in model output! "
                    "Prompt injection exfiltration attempt blocked."
                )

        # 2. Suspicious Exfiltration URL Check
        if self.block_suspicious_urls:
            for rx in self._url_regexes:
                for match in rx.finditer(llm_output):
                    found_url = match.group(0)
                    # Check against allowed domains
                    if allowed_domains and any(domain in found_url for domain in allowed_domains):
                        continue
                    suspicious_urls.append(found_url)
                    reasons.append(f"Suspicious egress exfiltration link detected: {found_url}")

        # 3. Secret Pattern Leak Check
        if self.block_sensitive_patterns:
            for rx in self._leak_regexes:
                if rx.search(llm_output):
                    sensitive_matches.append(rx.pattern)
                    reasons.append("Model output contains potential credential or API key leakage.")

        is_safe = not (canary_leaked or bool(suspicious_urls) or bool(sensitive_matches))

        if not is_safe:
            sanitized = (
                "[SECURITY INTERCEPTION]: InjectGuard blocked this response because it contained "
                "leaked security canary tokens or unauthorized data-exfiltration links."
            )
        else:
            sanitized = llm_output

        return EgressVerificationResult(
            is_safe=is_safe,
            sanitized_output=sanitized,
            canary_leaked=canary_leaked,
            suspicious_urls_found=suspicious_urls,
            sensitive_patterns_matched=sensitive_matches,
            reasons=reasons,
        )
