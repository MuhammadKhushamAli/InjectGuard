"""Evaluation runner for InjectGuard firewall.

Benchmarks individual layers and the combined pipeline:
- Layer 1: Sanitiser
- Layer 2: Heuristic & Rule Detector
- Layer 3: ML Classifiers (TF-IDF + Logistic Regression, DistilBERT)
- Combined Multi-Layer Pipeline

Reports for each layer:
- Accuracy, Precision, Recall, F1-Score
- False Positive Rate (FPR)
- Average added latency (in milliseconds)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from injectguard.detectors.classifier import DistilBertClassifier, LogisticRegressionClassifier
from injectguard.detectors.rules import RuleDetector
from injectguard.detectors.sanitiser import Sanitiser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTACKS_FILE = PROJECT_ROOT / "eval" / "datasets" / "attacks" / "redteam_attacks.jsonl"
BENIGN_FILE = PROJECT_ROOT / "eval" / "datasets" / "benign" / "benign_documents.jsonl"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def load_dataset() -> list[dict[str, Any]]:
    dataset = []
    with open(ATTACKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    with open(BENIGN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    return dataset


def evaluate_layer(
    name: str,
    predict_fn,
    dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """Benchmark a single detector layer across all samples."""
    y_true = []
    y_pred = []
    latencies_ms = []

    for item in dataset:
        text = item["text"]
        label = item["label"]  # 1 = attack, 0 = benign

        start_t = time.perf_counter()
        pred = predict_fn(text)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        y_true.append(label)
        y_pred.append(1 if pred else 0)
        latencies_ms.append(elapsed_ms)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / max(1, (fp + tn))
    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))

    return {
        "layer": name,
        "total_samples": len(dataset),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "avg_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
    }


def run_benchmark(include_distilbert: bool = False):
    print("=" * 70)
    print("InjectGuard Red-Team Attack Evaluation Suite")
    print("=" * 70)

    dataset = load_dataset()
    attacks_count = sum(1 for d in dataset if d["label"] == 1)
    benign_count = sum(1 for d in dataset if d["label"] == 0)
    print(f"Total benchmark samples: {len(dataset)} ({attacks_count} attacks, {benign_count} benign)\n")

    # Initialize detectors
    sanitiser = Sanitiser()
    rules = RuleDetector()
    lr = LogisticRegressionClassifier()
    distilbert = DistilBertClassifier() if include_distilbert else None

    # Layer 1: Sanitiser (checks if text was obfuscated / had hidden payloads)
    def pred_sanitiser(text: str) -> bool:
        res = sanitiser.sanitise(text)
        return res.was_obfuscated

    # Layer 2: Rule Detector
    def pred_rules(text: str) -> bool:
        clean = sanitiser.sanitise(text).cleaned_text
        return rules.predict_score(clean) >= 0.5

    # Layer 3: ML Logistic Regression
    def pred_lr(text: str) -> bool:
        clean = sanitiser.sanitise(text).cleaned_text
        return lr.predict_score(clean) >= 0.5

    # Combined Multi-Layer Pipeline
    def pred_combined(text: str) -> bool:
        s_res = sanitiser.sanitise(text)
        if s_res.was_obfuscated:
            return True
        r_score = rules.predict_score(s_res.cleaned_text)
        if r_score >= 0.5:
            return True
        lr_score = lr.predict_score(s_res.cleaned_text)
        if lr_score >= 0.5:
            return True
        if distilbert:
            db_score = distilbert.predict_score(s_res.cleaned_text)
            if db_score >= 0.5:
                return True
        return False

    results = []
    print("[1/4] Benchmarking Layer 1 (Sanitiser)...")
    results.append(evaluate_layer("Layer 1: Sanitiser", pred_sanitiser, dataset))

    print("[2/4] Benchmarking Layer 2 (Rules & Heuristics)...")
    results.append(evaluate_layer("Layer 2: Rules & Heuristics", pred_rules, dataset))

    print("[3/4] Benchmarking Layer 3 (ML Logistic Regression)...")
    results.append(evaluate_layer("Layer 3: ML Logistic Regression", pred_lr, dataset))

    if include_distilbert and distilbert:
        print("[*] Benchmarking Layer 3 (DistilBERT)...")
        def pred_distilbert(text: str) -> bool:
            clean = sanitiser.sanitise(text).cleaned_text
            return distilbert.predict_score(clean) >= 0.5
        results.append(evaluate_layer("Layer 3: DistilBERT Transformer", pred_distilbert, dataset))

    print("[4/4] Benchmarking Combined Pipeline...")
    results.append(evaluate_layer("Combined Pipeline (All Layers)", pred_combined, dataset))

    # Print Formatted Markdown Table
    header = f"| {'Layer':<30} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'FPR':<6} | {'Avg Latency':<12} |"
    divider = f"|{'-'*32}|{'-'*10}|{'-'*11}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*14}|"
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(header)
    print(divider)
    for r in results:
        row = (
            f"| {r['layer']:<30} "
            f"| {r['accuracy']*100:>6.1f}% "
            f"| {r['precision']*100:>7.1f}% "
            f"| {r['recall']*100:>5.1f}% "
            f"| {r['f1']*100:>4.1f}% "
            f"| {r['fpr']*100:>4.1f}% "
            f"| {r['avg_latency_ms']:>8.3f} ms |"
        )
        print(row)
    print("=" * 80)

    # Save results to json
    report_file = REPORTS_DIR / "benchmark_results.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    run_benchmark(include_distilbert=False)
