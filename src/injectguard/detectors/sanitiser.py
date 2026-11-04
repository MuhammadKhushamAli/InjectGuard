"""Layer 1: Sanitiser for InjectGuard.

Implements section 4.1.1 of the proposal:
"Normalises Unicode, strips invisible characters and hidden PDF text,
and decodes common encodings so that obfuscated payloads become visible."
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from pydantic import BaseModel, Field

# Zero-width, invisible, and format control Unicode characters
# Attackers insert these between letters (e.g. "i\u200Bgn\u200Core") to bypass naive string matching
INVISIBLE_CHARS_REGEX = re.compile(
    r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u00AD\u2060\u2061\u2062\u2063\u2064\u180E]"
)

# Common homoglyph mappings (Cyrillic/Greek lookalikes often used to spoof Latin letters)
HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
}

# Regex to detect Base64 encoded payload blocks
BASE64_CANDIDATE_REGEX = re.compile(
    r"\b[A-Za-z0-9+/]{8,}={0,2}\b"
)

# Regex to detect Hex encoded payload blocks (e.g., 69676e6f726520...)
HEX_CANDIDATE_REGEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){8,}\b")


class SanitisedOutput(BaseModel):
    """Output from the Layer 1 Sanitiser."""
    cleaned_text: str = Field(..., description="Normalized and de-obfuscated text")
    decoded_payloads: list[str] = Field(default_factory=list, description="Any hidden payloads unmasked")
    stripped_invisible_count: int = Field(0, description="Count of zero-width characters removed")
    homoglyphs_replaced_count: int = Field(0, description="Count of homoglyphs replaced")
    was_obfuscated: bool = Field(False, description="True if any evasion tricks were detected and de-masked")


class Sanitiser:
    """Layer 1: Normalizes Unicode, strips zero-width chars, decodes base64/hex."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        strip_invisible: bool = True,
        replace_homoglyphs: bool = True,
        decode_encodings: bool = True,
    ) -> None:
        self.normalize_unicode = normalize_unicode
        self.strip_invisible = strip_invisible
        self.replace_homoglyphs = replace_homoglyphs
        self.decode_encodings = decode_encodings

    def sanitise(self, text: str) -> SanitisedOutput:
        """Sanitise text and reveal any hidden obfuscated payloads."""
        if not text:
            return SanitisedOutput(cleaned_text="")

        working_text = str(text)
        stripped_count = 0
        homoglyph_count = 0
        decoded_payloads: list[str] = []

        # 1. Strip zero-width & invisible characters
        if self.strip_invisible:
            matches = INVISIBLE_CHARS_REGEX.findall(working_text)
            stripped_count = len(matches)
            working_text = INVISIBLE_CHARS_REGEX.sub("", working_text)

        # 2. Unicode Normalization (NFKC - Normalization Form Compatibility Composition)
        # Translates full-width chars, ligature characters, and non-standard forms to canonical Latin
        if self.normalize_unicode:
            working_text = unicodedata.normalize("NFKC", working_text)

        # 3. Homoglyph Normalization
        if self.replace_homoglyphs:
            homoglyph_replaced = []
            for char in working_text:
                if char in HOMOGLYPH_MAP:
                    homoglyph_replaced.append(HOMOGLYPH_MAP[char])
                    homoglyph_count += 1
                else:
                    homoglyph_replaced.append(char)
            working_text = "".join(homoglyph_replaced)

        # 4. Decode common obfuscated encodings (Base64 & Hex)
        if self.decode_encodings:
            # Base64 decoding scan
            for match in BASE64_CANDIDATE_REGEX.finditer(working_text):
                candidate = match.group(0)
                # Ensure length is multiple of 4
                padded = candidate + "=" * ((4 - len(candidate) % 4) % 4)
                try:
                    decoded_bytes = base64.b64decode(padded, validate=True)
                    decoded_str = decoded_bytes.decode("utf-8")
                    if any(c.isalpha() for c in decoded_str) and all(
                        c.isprintable() or c in "\r\n\t " for c in decoded_str
                    ):
                        if len(decoded_str.strip()) >= 5:
                            decoded_payloads.append(decoded_str.strip())
                except Exception:
                    pass

            # Hex decoding scan
            for match in HEX_CANDIDATE_REGEX.finditer(working_text):
                candidate = match.group(0)
                try:
                    decoded_bytes = binascii.unhexlify(candidate)
                    decoded_str = decoded_bytes.decode("utf-8")
                    if any(c.isalpha() for c in decoded_str) and all(
                        c.isprintable() or c in "\r\n\t " for c in decoded_str
                    ):
                        if len(decoded_str.strip()) >= 5:
                            decoded_payloads.append(decoded_str.strip())
                except Exception:
                    pass

            # If any hidden payloads were extracted, append them visibly to the cleaned text
            # so the downstream Rule and ML classifiers can analyze the hidden payload!
            if decoded_payloads:
                unmasked_block = " [UNMASKED_PAYLOAD: " + " | ".join(decoded_payloads) + "]"
                working_text = working_text + unmasked_block

        was_obfuscated = bool(stripped_count > 0 or homoglyph_count > 0 or decoded_payloads)

        return SanitisedOutput(
            cleaned_text=working_text.strip(),
            decoded_payloads=decoded_payloads,
            stripped_invisible_count=stripped_count,
            homoglyphs_replaced_count=homoglyph_count,
            was_obfuscated=was_obfuscated,
        )
