"""Stage 4: Token boundary analysis for split-word and stuffing detection."""
from __future__ import annotations

import re

from injection_guard.types import TokenSignals

__all__ = ["TokenBoundaryDetector"]

# Injection-related keywords to look for after reconstruction.
_INJECTION_KEYWORDS: list[str] = [
    "ignore",
    "previous",
    "instructions",
    "disregard",
    "override",
    "system",
    "prompt",
    "pretend",
    "jailbreak",
]

# Characters that may be used to split words (whitespace, zero-width, etc.).
_SPLIT_CHARS_RE = re.compile(r"[\s\u200b\u200c\u200d\u00ad\ufeff]+")

# Simple word tokeniser for repetition analysis.
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


class TokenBoundaryDetector:
    """Detects word splitting attacks and prompt stuffing.

    This is Stage 4 of the preprocessor pipeline.
    """

    def analyze(self, text: str) -> TokenSignals:
        """Scan *text* for token-boundary anomalies.

        Args:
            text: The input string (preferably already Unicode-normalized).

        Returns:
            A ``TokenSignals`` dataclass.
        """
        reconstructed_keywords = self._detect_split_keywords(text)
        prompt_length_percentile = self._length_percentile(text)
        repetition_ratio = self._repetition_ratio(text)

        return TokenSignals(
            reconstructed_keywords=reconstructed_keywords,
            prompt_length_percentile=prompt_length_percentile,
            repetition_ratio=repetition_ratio,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_split_keywords(text: str) -> list[str]:
        """Reconstruct keywords that may have been split by whitespace/zero-width chars.

        Walks through the text, collapsing sequences of split characters and
        checking whether the resulting concatenation forms an injection keyword.
        """
        found: list[str] = []

        # Collapse splitting characters to produce a "joined" version, then
        # look for injection keywords in the lowered result.
        collapsed = _SPLIT_CHARS_RE.sub("", text).lower()

        for keyword in _INJECTION_KEYWORDS:
            if keyword in collapsed:
                # Confirm the keyword was actually split in the original text
                # (i.e. not just present as a normal word).
                pattern = _build_split_pattern(keyword)
                if pattern.search(text) and not re.search(
                    r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE
                ):
                    found.append(keyword)

        return found

    @staticmethod
    def _length_percentile(text: str) -> float:
        """Return a heuristic length percentile for *text*.

        Uses simple thresholds rather than a corpus distribution:
        - <= 200 chars  -> 0.3
        - <= 500 chars  -> 0.5
        - <= 2000 chars -> 0.7
        - > 2000 chars  -> 0.9
        """
        length = len(text)
        if length <= 200:
            return 0.3
        if length <= 500:
            return 0.5
        if length <= 2000:
            return 0.7
        return 0.9

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        """Compute ratio of repeated words to total words."""
        words = _WORD_RE.findall(text.lower())
        if not words:
            return 0.0
        unique = set(words)
        repeated = len(words) - len(unique)
        return repeated / len(words)


def _build_split_pattern(keyword: str) -> re.Pattern[str]:
    """Build a regex that matches *keyword* with optional split chars between letters."""
    parts = list(keyword)
    sep = r"[\s\u200b\u200c\u200d\u00ad\ufeff]+"
    pattern_str = sep.join(re.escape(ch) for ch in parts)
    return re.compile(pattern_str, re.IGNORECASE)
