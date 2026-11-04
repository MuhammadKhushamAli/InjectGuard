"""Integrity verification for InjectGuard assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


def verify_sha256(file_path: Path | str, expected_sha_path: Path | str) -> bool:
    """Verify that a file matches its corresponding SHA-256 hash file.

    Args:
        file_path: Path to the target file.
        expected_sha_path: Path to the file containing the expected hex digest.

    Returns:
        True if valid, raises ValueError if mismatch.
    """
    file_p = Path(file_path)
    sha_p = Path(expected_sha_path)

    if not file_p.exists():
        raise FileNotFoundError(f"Target file not found: {file_p}")
    if not sha_p.exists():
        raise FileNotFoundError(f"Checksum file not found: {sha_p}")

    with open(sha_p, "r", encoding="utf-8") as f:
        expected = f.read().strip()

    hasher = hashlib.sha256()
    with open(file_p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual = hasher.hexdigest()

    if actual.lower() != expected.lower():
        raise ValueError(
            f"Integrity check failed for {file_p.name}!\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )
    return True
