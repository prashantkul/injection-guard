"""Stage 6: Regex pattern matching for known injection patterns."""
from __future__ import annotations

import re

from injection_guard.types import RegexSignals

__all__ = ["RegexAnalyzer"]

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"you\s+are\s+now\s+(in\s+)?",
        r"pretend\s+(you\s+are|to\s+be)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"forget\s+(all\s+)?(previous|prior|above|your)",
        r"do\s+not\s+follow\s+(previous|prior|above|any)",
        r"skip\s+(all\s+)?instructions",
        r"override\s+(all\s+)?(safety|security|instructions)",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
        r"show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
    ]
]


class RegexAnalyzer:
    """Scans text for known injection regex patterns.

    This runs as Stage 6 of the preprocessor pipeline, contributing
    matched patterns to the signal vector and boosting risk_prior.
    """

    def __init__(
        self, *, extra_patterns: list[str] | None = None
    ) -> None:
        """Initialise with default + optional custom patterns.

        Args:
            extra_patterns: Additional regex patterns to compile and match.
        """
        self._patterns = list(_INJECTION_PATTERNS)
        if extra_patterns:
            self._patterns.extend(
                re.compile(p, re.IGNORECASE) for p in extra_patterns
            )

    def analyze(self, text: str) -> RegexSignals:
        """Scan *text* against all compiled injection patterns.

        Args:
            text: Normalized prompt text.

        Returns:
            RegexSignals with all matched patterns and texts.
        """
        matched_patterns: list[str] = []
        matched_texts: list[str] = []

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                matched_patterns.append(pattern.pattern)
                matched_texts.append(match.group())

        return RegexSignals(
            matched_patterns=matched_patterns,
            matched_texts=matched_texts,
            match_count=len(matched_patterns),
        )
