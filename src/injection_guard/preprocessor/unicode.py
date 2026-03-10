"""Stage 1: Unicode normalization and anomaly detection."""
from __future__ import annotations

import re
import unicodedata

from injection_guard.types import UnicodeSignals

__all__ = ["UnicodeNormalizer"]

# Zero-width characters to strip and count.
_ZERO_WIDTH_CHARS: set[int] = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x00AD,  # SOFT HYPHEN
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
}

# Bidirectional override / embedding / isolate codepoints.
_BIDI_CODEPOINTS: set[int] = {
    *range(0x202A, 0x202F),  # LRE, RLE, PDF, LRO, RLO
    *range(0x2066, 0x206A),  # LRI, RLI, FSI, PDI
}

# Script categories used for script-mixing detection.
_LATIN_SCRIPTS = {"LATIN"}
_CYRILLIC_SCRIPTS = {"CYRILLIC"}

_WORD_RE = re.compile(r"[^\s]+")


class UnicodeNormalizer:
    """Applies NFKC normalization and extracts Unicode-based signals.

    This is Stage 1 of the preprocessor pipeline.  It detects homoglyphs,
    zero-width characters, bidirectional overrides, and script mixing.
    """

    def analyze(self, text: str) -> tuple[str, UnicodeSignals]:
        """Normalize *text* and return (normalized_text, signals).

        Args:
            text: The raw input string.

        Returns:
            A tuple of the NFKC-normalized string (with zero-width and bidi
            characters stripped) and a ``UnicodeSignals`` dataclass.
        """
        nfkc = unicodedata.normalize("NFKC", text)

        # --- homoglyph detection ---
        homoglyph_count = self._count_homoglyphs(text, nfkc)

        # --- zero-width characters ---
        zero_width_count = 0
        suspicious_codepoints: list[str] = []
        cleaned_chars: list[str] = []
        for ch in nfkc:
            cp = ord(ch)
            if cp in _ZERO_WIDTH_CHARS:
                zero_width_count += 1
                suspicious_codepoints.append(f"U+{cp:04X}")
            elif cp in _BIDI_CODEPOINTS:
                # counted separately below; strip from output
                pass
            else:
                cleaned_chars.append(ch)

        # --- bidi overrides ---
        bidi_override_count = 0
        for ch in nfkc:
            cp = ord(ch)
            if cp in _BIDI_CODEPOINTS:
                bidi_override_count += 1
                suspicious_codepoints.append(f"U+{cp:04X}")

        normalized = "".join(cleaned_chars)

        # --- script mixing ---
        script_mixing = self._detect_script_mixing(normalized)

        # --- edit distance ---
        normalization_edit_distance = self._edit_distance(text, nfkc)

        signals = UnicodeSignals(
            homoglyph_count=homoglyph_count,
            zero_width_count=zero_width_count,
            bidi_override_count=bidi_override_count,
            normalization_edit_distance=normalization_edit_distance,
            script_mixing=script_mixing,
            suspicious_codepoints=suspicious_codepoints,
        )
        return normalized, signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Known confusable mappings: non-ASCII codepoints that look like ASCII.
    _CONFUSABLES: set[int] = {
        0x0406, 0x0410, 0x0412, 0x0415, 0x041A, 0x041C, 0x041D, 0x041E,
        0x0420, 0x0421, 0x0422, 0x0425,  # Cyrillic uppercase
        0x0430, 0x0435, 0x043E, 0x0440, 0x0441, 0x0443, 0x0445, 0x0456,
        # Cyrillic lowercase
        0xFF21, 0xFF22, 0xFF23, 0xFF24, 0xFF25,  # Fullwidth Latin uppercase
        0xFF41, 0xFF42, 0xFF43, 0xFF44, 0xFF45,  # Fullwidth Latin lowercase
    }

    @staticmethod
    def _count_homoglyphs(original: str, nfkc: str) -> int:
        """Count characters that are potential homoglyphs.

        A character is a homoglyph if it changes under NFKC normalization,
        or if it is a known confusable codepoint (e.g., Cyrillic letters
        that visually resemble Latin ones).
        """
        count = 0
        for orig_ch, norm_ch in zip(original, nfkc):
            if orig_ch != norm_ch:
                count += 1
            elif ord(orig_ch) in UnicodeNormalizer._CONFUSABLES:
                count += 1
        # If lengths differ, remaining chars also count.
        count += abs(len(original) - len(nfkc))
        return count

    @staticmethod
    def _detect_script_mixing(text: str) -> bool:
        """Return True if any word contains both Latin and Cyrillic chars."""
        for match in _WORD_RE.finditer(text):
            word = match.group()
            has_latin = False
            has_cyrillic = False
            for ch in word:
                try:
                    script = unicodedata.name(ch, "").split()[0]
                except (ValueError, IndexError):
                    continue
                if script in _LATIN_SCRIPTS:
                    has_latin = True
                elif script in _CYRILLIC_SCRIPTS:
                    has_cyrillic = True
                if has_latin and has_cyrillic:
                    return True
        return False

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Compute character-level Levenshtein distance between *a* and *b*."""
        if a == b:
            return 0
        len_a, len_b = len(a), len(b)
        if len_a == 0:
            return len_b
        if len_b == 0:
            return len_a

        # Use two-row DP to save memory.
        prev = list(range(len_b + 1))
        curr = [0] * (len_b + 1)
        for i in range(1, len_a + 1):
            curr[0] = i
            for j in range(1, len_b + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost, # substitution
                )
            prev, curr = curr, prev
        return prev[len_b]
