"""Unit tests for Multi-Modal Extractors."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from injectguard.extractors.router import extract_from_audio, extract_from_image, extract_from_text


def test_text_extractor():
    raw = "   Sample input text query   "
    extracted = extract_from_text(raw)
    assert extracted == "Sample input text query"


def test_image_extractor(tmp_path: Path):
    # Create simple image in tmp_path
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    extracted = extract_from_image(img_path)
    assert isinstance(extracted, str)


def test_audio_extractor():
    extracted = extract_from_audio("sample_voice.wav", openai_client=None)
    assert "AUDIO_TRANSCRIPTION" in extracted
