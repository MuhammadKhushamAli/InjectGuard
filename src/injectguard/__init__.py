"""InjectGuard Public Library API."""

from __future__ import annotations

from injectguard.audit import AuditLogger, AuditRecord
from injectguard.detectors.cascade import MultiModalLangGraphGuard
from injectguard.detectors.classifier import DistilBertClassifier, LogisticRegressionClassifier
from injectguard.detectors.judge import LLMJudge
from injectguard.detectors.rules import RuleDetector
from injectguard.detectors.sanitiser import Sanitiser, SanitisedOutput
from injectguard.models import ExtractedContent, InspectionDecision
from injectguard.runtime.canary import CanaryInjectionResult, CanaryManager
from injectguard.runtime.egress import EgressGuard, EgressVerificationResult


class InjectGuard(MultiModalLangGraphGuard):
    """Full-featured Prompt Injection Firewall SDK.

    Includes:
    - Layer 1: Sanitiser (Unicode normalization, zero-width stripping, base64 decoding)
    - Layer 2: Heuristic & Rule Detector
    - Layer 3: ML Classifiers (TF-IDF + Logistic Regression, DistilBERT)
    - Layer 4: Runtime Guard (Canary token manager & Egress output scanner)
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        use_distilbert: bool = True,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        super().__init__(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            use_distilbert=use_distilbert,
            audit_logger=audit_logger,
        )
        self.canary_manager = CanaryManager()
        self.egress_guard = EgressGuard()

    def protect_system_prompt(
        self,
        base_system_prompt: str,
        canary_token: str | None = None,
    ) -> CanaryInjectionResult:
        """Inject a high-entropy canary token into the system prompt."""
        return self.canary_manager.inject_canary(
            base_system_prompt=base_system_prompt,
            canary_token=canary_token,
        )

    def verify_output(
        self,
        llm_output: str,
        canary_token: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> EgressVerificationResult:
        """Scan LLM response before egress to prevent canary or context leaks."""
        result = self.egress_guard.verify_output(
            llm_output=llm_output,
            canary_token=canary_token,
            allowed_domains=allowed_domains,
        )

        if not result.is_safe:
            self.audit_logger.log(
                event_type="EGRESS_BLOCKED",
                modality="text",
                decision=InspectionDecision(
                    score=1.0,
                    is_accepted=False,
                    reason="; ".join(result.reasons),
                ),
                raw_content=llm_output,
            )

        return result


__all__ = [
    "InjectGuard",
    "MultiModalLangGraphGuard",
    "CanaryManager",
    "CanaryInjectionResult",
    "EgressGuard",
    "EgressVerificationResult",
    "Sanitiser",
    "SanitisedOutput",
    "InspectionDecision",
    "ExtractedContent",
    "LogisticRegressionClassifier",
    "DistilBertClassifier",
    "RuleDetector",
    "LLMJudge",
    "AuditLogger",
    "AuditRecord",
]
