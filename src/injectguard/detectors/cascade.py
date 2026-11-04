"""LangGraph-powered Multi-Modal Inspection Graph for InjectGuard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict
from langgraph.graph import END, StateGraph

from injectguard.audit import AuditLogger
from injectguard.detectors.classifier import DistilBertClassifier, LogisticRegressionClassifier
from injectguard.detectors.judge import LLMJudge
from injectguard.detectors.rules import RuleDetector
from injectguard.detectors.sanitiser import Sanitiser, SanitisedOutput
from injectguard.extractors.router import extract_from_audio, extract_from_image, extract_from_text
from injectguard.models import InspectionDecision


class GuardState(TypedDict, total=False):
    """LangGraph state representation."""
    raw_input: Any
    modality: Literal["text", "audio", "image"]
    extracted_text: str
    sanitised: SanitisedOutput
    scores: dict[str, float]
    decision: InspectionDecision


class MultiModalLangGraphGuard:
    """LangGraph workflow to route, extract, sanitise, detect, and judge inputs."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        use_distilbert: bool = True,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.sanitiser = Sanitiser()
        self.lr_classifier = LogisticRegressionClassifier()
        self.rule_detector = RuleDetector()
        self.use_distilbert = use_distilbert
        self.distilbert_classifier = DistilBertClassifier() if use_distilbert else None
        self.judge = LLMJudge(api_key=openai_api_key, model=openai_model)
        self.audit_logger = audit_logger or AuditLogger()
        self.graph = self._build_graph()

    # --- Node Definitions ---

    def _route_modality(self, state: GuardState) -> GuardState:
        """Determines the modality of raw_input by checking types, magic bytes, or file extension."""
        modality = state.get("modality")
        if modality:
            return {"modality": modality}

        raw = state.get("raw_input")

        # 1. Check Python object types (PIL Image, etc.)
        if hasattr(raw, "format") and hasattr(raw, "size"):
            return {"modality": "image"}

        # 2. Check binary magic bytes
        if isinstance(raw, (bytes, bytearray)):
            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff"):
                return {"modality": "image"}
            if raw.startswith(b"RIFF") or raw.startswith(b"ID3") or raw.startswith(b"OggS"):
                return {"modality": "audio"}

        # 3. Check file path string / Path
        if isinstance(raw, (str, Path)):
            raw_str = str(raw).strip()
            lower = raw_str.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
                return {"modality": "image"}
            if lower.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")):
                return {"modality": "audio"}

        # Default to text
        return {"modality": "text"}

    def _extract_text_node(self, state: GuardState) -> GuardState:
        raw = state["raw_input"]
        return {"extracted_text": extract_from_text(raw)}

    def _extract_image_node(self, state: GuardState) -> GuardState:
        raw = state["raw_input"]
        return {"extracted_text": extract_from_image(raw)}

    def _extract_audio_node(self, state: GuardState) -> GuardState:
        raw = state["raw_input"]
        return {"extracted_text": extract_from_audio(raw, openai_client=self.judge.client)}

    def _sanitise_node(self, state: GuardState) -> GuardState:
        """Layer 1: Normalises Unicode, strips zero-width chars, decodes hidden payloads."""
        text = state["extracted_text"]
        sanitised_result = self.sanitiser.sanitise(text)
        return {
            "sanitised": sanitised_result,
            "extracted_text": sanitised_result.cleaned_text,
        }

    def _run_models_node(self, state: GuardState) -> GuardState:
        """Runs the detection models against the sanitised text."""
        text = state["extracted_text"]
        sanitised = state.get("sanitised")
        scores: dict[str, float] = {}

        # If obfuscation / evasion tricks were detected, record an evasion penalty
        if sanitised and sanitised.was_obfuscated:
            scores["obfuscation_detected"] = 1.0

        # 1. Rule & heuristic detector score
        scores["rule_score"] = self.rule_detector.predict_score(text)

        # 2. Logistic Regression score
        scores["logistic_regression_score"] = self.lr_classifier.predict_score(text)

        # 3. DistilBERT score (if enabled)
        if self.distilbert_classifier:
            try:
                scores["distilbert_score"] = self.distilbert_classifier.predict_score(text)
            except Exception:
                pass

        return {"scores": scores}

    def _llm_judge_node(self, state: GuardState) -> GuardState:
        """LLM aggregates all model scores into the final Pydantic InspectionDecision."""
        decision = self.judge.judge(
            extracted_text=state["extracted_text"],
            modality=state["modality"],
            model_scores=state.get("scores", {}),
        )
        return {"decision": decision}

    # --- Graph Assembly ---

    def _build_graph(self):
        workflow = StateGraph(GuardState)

        # Nodes
        workflow.add_node("router", self._route_modality)
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("extract_image", self._extract_image_node)
        workflow.add_node("extract_audio", self._extract_audio_node)
        workflow.add_node("sanitise", self._sanitise_node)
        workflow.add_node("run_models", self._run_models_node)
        workflow.add_node("llm_judge", self._llm_judge_node)

        # Edges
        workflow.set_entry_point("router")

        def route_by_modality(state: GuardState):
            modality = state.get("modality", "text")
            if modality == "image":
                return "extract_image"
            elif modality == "audio":
                return "extract_audio"
            return "extract_text"

        workflow.add_conditional_edges(
            "router",
            route_by_modality,
            {
                "extract_text": "extract_text",
                "extract_image": "extract_image",
                "extract_audio": "extract_audio",
            },
        )

        workflow.add_edge("extract_text", "sanitise")
        workflow.add_edge("extract_image", "sanitise")
        workflow.add_edge("extract_audio", "sanitise")
        workflow.add_edge("sanitise", "run_models")
        workflow.add_edge("run_models", "llm_judge")
        workflow.add_edge("llm_judge", END)

        return workflow.compile()

    def inspect(
        self,
        raw_input: Any,
        modality: Literal["text", "audio", "image"] | None = None,
    ) -> InspectionDecision:
        """Inspect any multi-modal input and return the 3-field InspectionDecision."""
        initial_state: GuardState = {
            "raw_input": raw_input,
            "modality": modality,
            "scores": {},
        }
        final_state = self.graph.invoke(initial_state)
        decision = final_state["decision"]

        # Audit log decision
        event_type = "INCORPORATED" if decision.is_accepted else "QUARANTINED"
        self.audit_logger.log(
            event_type=event_type,
            modality=final_state.get("modality", "text"),
            decision=decision,
            raw_content=raw_input,
        )

        return decision

    def filter_chunks(
        self,
        chunks: list[Any],
        modality: Literal["text", "audio", "image"] | None = None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Inspect a collection of retrieved chunks or user inputs.

        Rejects and deletes any poisoned chunk, while returning only the
        safe chunks to incorporate into RAG context.
        """
        safe_chunks: list[Any] = []
        rejected_records: list[dict[str, Any]] = []

        for chunk in chunks:
            decision = self.inspect(chunk, modality=modality)
            if decision.is_accepted:
                safe_chunks.append(chunk)
            else:
                rejected_records.append({
                    "chunk": chunk,
                    "score": decision.score,
                    "reason": decision.reason,
                })

        return safe_chunks, rejected_records
