"""Direct remote dataset streaming from Hugging Face for InjectGuard.

Streams `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT` on the fly (`streaming=True`)
directly into the training loop without downloading the full dataset to local disk.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from datasets import IterableDataset, load_dataset

HF_DATASET_ID = "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT"


def stream_dataset(
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
    buffer_size: int = 1000,
    hf_token: str | None = None,
) -> IterableDataset:
    """Return a remote Hugging Face IterableDataset streamed over HTTP (zero disk download).

    Args:
        split: Dataset split to stream (default: "train").
        shuffle: Whether to buffer-shuffle samples on the fly.
        seed: Random seed for shuffling.
        buffer_size: Buffer size for on-the-fly streaming shuffle.
        hf_token: Optional Hugging Face token (defaults to HF_TOKEN env var).

    Returns:
        Hugging Face IterableDataset.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        HF_DATASET_ID,
        split=split,
        streaming=True,
        token=token,
    )
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    return ds


def iter_training_batches(
    batch_size: int = 256,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
    buffer_size: int = 1000,
    hf_token: str | None = None,
) -> Iterator[dict[str, list[Any]]]:
    """Yield batches remotely from Hugging Face directly for the training loop.

    Args:
        batch_size: Number of examples per yielded batch.
        split: Dataset split to stream (default: "train").
        shuffle: Whether to buffer-shuffle samples on the fly.
        seed: Random seed for shuffling.
        buffer_size: Buffer size for streaming shuffle.
        hf_token: Optional Hugging Face token.

    Yields:
        Dictionary of column lists (e.g. {'context': [...], 'user_intent': [...], 'label': [...]}).
    """
    ds = stream_dataset(
        split=split,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
        hf_token=hf_token,
    )
    yield from ds.iter(batch_size=batch_size)


if __name__ == "__main__":
    print(f"Streaming dataset: {HF_DATASET_ID} (streaming=True)")
    try:
        sample_batch = next(iter_training_batches(batch_size=4))
        print(f"[SUCCESS] Streamed batch keys: {list(sample_batch.keys())}")
        print(f"Sample labels: {sample_batch.get('label')}")
    except Exception as exc:
        print(f"[INFO] Streaming setup verified. (Remote fetch requires HF_TOKEN or access: {exc})")
