"""Data preparation and streaming from Hugging Face for InjectGuard.

Directly streams or loads prompt injection datasets from Hugging Face
without requiring local manual downloads.
"""

from __future__ import annotations

import os
from typing import Any, Iterator
from datasets import IterableDataset, Dataset, load_dataset

# Public prompt injection dataset on Hugging Face (no token required)
DEFAULT_HF_DATASET = "deepset/prompt-injections"


def get_dataset(
    dataset_name: str = DEFAULT_HF_DATASET,
    split: str = "train",
    streaming: bool = False,
    hf_token: str | None = None,
) -> Dataset | IterableDataset:
    """Fetch or stream a dataset directly from Hugging Face Hub.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split ('train', 'test', etc.).
        streaming: If True, streams examples over HTTP on-the-fly.
        hf_token: Optional token for gated/private datasets.

    Returns:
        Hugging Face Dataset or IterableDataset.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        token=token,
    )


def iter_training_batches(
    dataset_name: str = DEFAULT_HF_DATASET,
    batch_size: int = 128,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
    buffer_size: int = 1000,
    hf_token: str | None = None,
) -> Iterator[dict[str, list[Any]]]:
    """Yield streamed batches directly from Hugging Face for training/eval."""
    ds = get_dataset(
        dataset_name=dataset_name,
        split=split,
        streaming=True,
        hf_token=hf_token,
    )
    if shuffle and hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    yield from ds.iter(batch_size=batch_size)


if __name__ == "__main__":
    print(f"Connecting to Hugging Face dataset: '{DEFAULT_HF_DATASET}'...")
    sample_batch = next(iter_training_batches(batch_size=3))
    print("[SUCCESS] Successfully streamed sample batch:")
    for text, label in zip(sample_batch["text"], sample_batch["label"]):
        print(f"  - Label {label}: {text[:80]}...")
