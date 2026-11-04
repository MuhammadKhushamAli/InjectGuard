"""ML-based prompt injection classifiers for InjectGuard.

Includes:
1. LogisticRegressionClassifier (Baseline lightweight TF-IDF model)
2. DistilBertClassifier (Deep learning contextual transformer model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib

from injectguard.integrity import verify_sha256

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL_FILE = DEFAULT_DATA_DIR / "lr.joblib"
DEFAULT_SHA_FILE = DEFAULT_DATA_DIR / "lr.sha256"
DEFAULT_DISTILBERT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "distilbert"


class LogisticRegressionClassifier:
    """Lightweight ML detector using TF-IDF and Logistic Regression."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_FILE,
        sha_path: Path | str = DEFAULT_SHA_FILE,
        threshold: float = 0.5,
        verify_integrity: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.sha_path = Path(sha_path)
        self.threshold = threshold

        if verify_integrity and self.sha_path.exists():
            verify_sha256(self.model_path, self.sha_path)

        self.model = joblib.load(self.model_path)

    def predict_score(self, text: str) -> float:
        """Return the probability score of injection (0.0 to 1.0)."""
        proba = self.model.predict_proba([text])[0][1]
        return float(proba)

    def predict(self, text: str) -> dict[str, Any]:
        """Classify whether the text is an indirect prompt injection."""
        score = self.predict_score(text)
        is_injection = score >= self.threshold
        return {
            "score": score,
            "is_injection": is_injection,
            "threshold": self.threshold,
            "model_type": "logistic_regression",
        }


class DistilBertClassifier:
    """Advanced contextual ML detector using fine-tuned DistilBERT transformer."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_DISTILBERT_DIR,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.threshold = threshold

        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            # Fallback to pre-trained HF model or base weights if local checkpoints are not yet fine-tuned
            model_target = "distilbert-base-uncased"
        else:
            model_target = str(self.model_dir)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_target)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_target)
        self.model.to(self.device)
        self.model.eval()

    def predict_score(self, text: str) -> float:
        """Compute the softmax probability of the prompt injection class."""
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            # Probability of label 1 (INJECTION)
            score = probabilities[0][1].item()
        return float(score)

    def predict(self, text: str) -> dict[str, Any]:
        """Classify whether the text is an indirect prompt injection."""
        score = self.predict_score(text)
        is_injection = score >= self.threshold
        return {
            "score": score,
            "is_injection": is_injection,
            "threshold": self.threshold,
            "model_type": "distilbert",
        }
