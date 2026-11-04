"""Fine-tune DistilBERT for Prompt Injection Detection in InjectGuard.

Directly downloads and fine-tunes DistilBERT from Hugging Face on the injection dataset.
Saves the fine-tuned model weights and tokenizer into `models/distilbert`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert"
BASE_MODEL_NAME = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    """Compute precision, recall, f1, and accuracy for evaluation."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    prec = precision_metric.compute(predictions=preds, references=labels)["precision"]
    rec = recall_metric.compute(predictions=preds, references=labels)["recall"]

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


def train_distilbert(
    dataset_name: str = "deepset/prompt-injections",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model: str = BASE_MODEL_NAME,
    num_train_epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    hf_token: str | None = None,
):
    """Fine-tune DistilBERT directly from Hugging Face."""
    print("=" * 65)
    print("InjectGuard - DistilBERT Fine-Tuning Pipeline")
    print(f"Base Model:  {base_model}")
    print(f"Dataset:     {dataset_name}")
    print(f"Output Path: {output_dir}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Tokenizer & Model
    print("\n[1/5] Loading base DistilBERT tokenizer and classification model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "BENIGN", 1: "INJECTION"},
        label2id={"BENIGN": 0, "INJECTION": 1},
        token=hf_token,
    )

    # 2. Load Dataset directly from Hugging Face
    print("\n[2/5] Loading dataset from Hugging Face Hub...")
    raw_ds = load_dataset(dataset_name, token=hf_token)

    train_split = raw_ds["train"]
    test_split = raw_ds["test"] if "test" in raw_ds else train_split

    text_col = "text" if "text" in train_split.column_names else "context"

    def tokenize_fn(examples):
        return tokenizer(examples[text_col], truncation=True, max_length=256)

    print("\n[3/5] Tokenizing dataset...")
    tokenized_train = train_split.map(tokenize_fn, batched=True)
    tokenized_test = test_split.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 4. Train
    print("\n[4/5] Training model...")
    trainer.train()

    # 5. Save best model and tokenizer
    print("\n[5/5] Saving fine-tuned DistilBERT model & tokenizer...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved successfully to: {output_dir}")

    # Evaluate
    print("\nFinal Test Metrics:")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k:20}: {v:.4f}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for prompt injection detection")
    parser.add_argument("--dataset", type=str, default="deepset/prompt-injections")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_distilbert(
        dataset_name=args.dataset,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
    )
