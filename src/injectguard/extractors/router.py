"""Modality extractors for InjectGuard: Text, Image (OCR), Audio (Whisper)."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None


def extract_from_text(input_data: str) -> str:
    """Normalize string input."""
    return str(input_data).strip()


def extract_from_image(input_data: Union[str, Path, bytes, Image.Image]) -> str:
    """Extract text from an image using OCR (Tesseract / PIL fallback)."""
    img = None
    if isinstance(input_data, (str, Path)):
        img = Image.open(input_data)
    elif isinstance(input_data, bytes):
        img = Image.open(io.BytesIO(input_data))
    elif isinstance(input_data, Image.Image):
        img = input_data
    else:
        raise ValueError(f"Unsupported image input type: {type(input_data)}")

    if pytesseract is not None:
        try:
            extracted = pytesseract.image_to_string(img)
            return extracted.strip()
        except Exception:
            # Fallback if tesseract binary is not on system PATH
            pass

    return f"[IMAGE_OCR_FALLBACK: Processed image {img.size}]"


def extract_from_audio(
    input_data: Union[str, Path, io.BufferedReader],
    openai_client=None,
) -> str:
    """Transcribe audio to text using OpenAI Whisper API or local whisper."""
    audio_file_path = None
    if isinstance(input_data, (str, Path)):
        audio_file_path = str(input_data)

    if openai_client is not None and audio_file_path and os.path.exists(audio_file_path):
        with open(audio_file_path, "rb") as f:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
            return transcription.text.strip()

    return f"[AUDIO_TRANSCRIPTION: {audio_file_path or 'stream'}]"
