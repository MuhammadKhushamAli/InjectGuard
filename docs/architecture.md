# InjectGuard: System Architecture & Design Specification

**CS3002 Information Security — Fall 2026 • Section 7F • FAST-NUCES, Lahore**  
**Team Members:**  
- Muhammad Khusham Ali (23L-0865)  
- M. Ubaidullah (23L-2552)  
- Ahmad Hassan (23L-0861)  

---

## 1. System Overview

InjectGuard is an independent, model-agnostic security proxy and Python library designed to protect Retrieval-Augmented Generation (RAG) chatbots and LLM applications from indirect prompt injection, data exfiltration, and context poisoning.

Rather than relying on prompt-level system instructions alone, InjectGuard enforces **defense-in-depth** across 4 independent security layers before retrieved content ever reaches the target LLM, plus an output runtime guard before responses are returned to the user.

```
                              INCOMING CONTENT / RETRIEVED CHUNKS
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     Multi-Modal Router Node      │
                              │   (Text, OCR Image, Whisper)     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  LAYER 1: Sanitiser & Decoder    │
                              │ (Unicode NFKC, Zero-Width, B64)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ LAYER 2: Rule & Heuristic Engine │
                              │   (Regex overrides, system tags) │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     LAYER 3: ML Classifiers      │
                              │ • TF-IDF + Logistic Regression   │
                              │ • Fine-tuned DistilBERT          │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     LLM Cumulative Judge         │
                              │  (Pydantic 3-Field Resolution)   │
                              └────────────────┬─────────────────┘
                                               │
                          is_accepted = True   │   is_accepted = False
                 ┌─────────────────────────────┴─────────────────────────────┐
                 ▼                                                           ▼
    ┌─────────────────────────┐                                 ┌─────────────────────────┐
    │  INCORPORATE INTO RAG   │                                 │   DROP & QUARANTINE     │
    │ Context built with safe │                                 │ Malicious chunk deleted │
    │ chunks + Canary Token   │                                 │ Written to Audit Log    │
    └────────────┬────────────┘                                 └─────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │     TARGET LLM CALL     │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ LAYER 4: Output Guard   │
    │ (Canary & Egress Check) │
    └────────────┬────────────┘
                 │
                 ▼
           SAFE RESPONSE
```

---

## 2. Multi-Layer Pipeline Architecture

### Layer 1: Sanitiser (`src/injectguard/detectors/sanitiser.py`)
- **Zero-Width Stripping**: Eliminates `\u200B`, `\u200C`, `\uFEFF`, and bidirectional control characters used by attackers to split malicious keywords.
- **Unicode NFKC Normalization**: Standardizes ligature characters, full-width fonts, and mathematical script variations.
- **Homoglyph Normalization**: Transliterates visually identical Cyrillic and Greek characters (e.g., `а`, `е`, `о`, `р`) into Latin ASCII.
- **Encoding Unmasking**: Detects and decodes hidden Base64 and Hexadecimal blocks, appending unmasked text so downstream ML and rules can inspect the secret payload.

### Layer 2: Rule & Heuristic Detector (`src/injectguard/detectors/rules.py`)
- Regex-based signature detection for direct override directives (`ignore all previous instructions`, `system override`, `disregard rules`).
- Role-play and jailbreak signatures (`DAN mode`, `developer mode`, `evil assistant`).
- System-tag injection (`<system>`, `[system]`).
- Exfiltration sink endpoints (`requestcatcher`, `webhook`, `pipedream`).

### Layer 3: Machine Learning Classifiers (`src/injectguard/detectors/classifier.py`)
- **Baseline Model (TF-IDF + Logistic Regression)**:
  - Trained directly on public Hugging Face injection datasets (`deepset/prompt-injections`).
  - Sublinear term-frequency with unigram and bigram features ($15,000$ max features).
  - Cryptographically protected via SHA-256 integrity checksum verification (`lr.sha256`).
- **Contextual Transformer (DistilBERT)**:
  - Fine-tuned sequence classification model using Hugging Face `transformers` and `Trainer`.

### Layer 4: Runtime Canary Guard & Egress Scanner (`src/injectguard/runtime/`)
- **Canary Manager (`canary.py`)**: Injects a high-entropy 128-bit secret token into the LLM system prompt alongside a non-disclosure security boundary.
- **Egress Guard (`egress.py`)**: Intercepts the LLM output before delivery. If the canary token appears in the output, an injection has succeeded in prompt leakage—the output is immediately blocked and intercepted. Also scans for suspicious external URLs and API key leaks.

---

## 3. Red-Team Benchmark Results

Evaluated over **440 samples** (220 red-team attacks across 5 categories + 220 matched benign documents):

| Layer | Accuracy | Precision | Recall (Detection Rate) | F1-Score | False Positive Rate | Added Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1: Sanitiser** | 57.7% | 80.4% | 20.4% | 32.6% | 5.0% | **0.036 ms** |
| **Layer 2: Rules & Heuristics** | 72.0% | 100.0% | 44.1% | 61.2% | **0.0%** | **0.053 ms** |
| **Layer 3: ML Logistic Regression** | 74.6% | 72.7% | 78.6% | 75.5% | 29.5% | **1.151 ms** |
| **Combined Pipeline (All Layers)** | **83.0%** | **76.4%** | **95.5%** | **84.9%** | 29.5% | **0.689 ms** |

> **Key Insight**: The Rule layer provides 0% false positives for known signatures at negligible latency ($0.053\text{ ms}$), while the ML and Sanitiser layers catch novel and obfuscated attacks, elevating total recall to **95.5%**.

---

## 4. Library Installation & Usage

### Installation
```bash
pip install -e .
```

### Basic RAG Filtering
```python
from injectguard import InjectGuard

guard = InjectGuard()

# Filter untrusted chunks (Drop & Quarantine Policy)
safe_chunks, rejected = guard.filter_chunks(retrieved_chunks)

# Incorporate only safe chunks into context
context = "\n".join(safe_chunks)
```

### Canary Guarding
```python
# 1. Protect prompt
res = guard.protect_system_prompt("You are a customer assistant.")
canary = res.canary_token

# 2. Verify model output
output_check = guard.verify_output(llm_response, canary_token=canary)
if output_check.is_safe:
    print(output_check.sanitized_output)
else:
    print("Alert: Canary exfiltration blocked!")
```
