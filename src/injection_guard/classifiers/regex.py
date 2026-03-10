"""Regex-based pre-filter classifier for fast injection detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from injection_guard.types import ClassifierResult, SignalVector

__all__ = ["RegexPrefilter"]

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
    ]
]


@dataclass
class RegexPrefilter:
    """Fast regex-based classifier that scans for known injection patterns.

    Attributes:
        name: Classifier identifier.
        latency_tier: Always ``"fast"`` for this classifier.
        weight: Weight used by the ensemble aggregator.
    """

    name: str = "regex-prefilter"
    latency_tier: str = "fast"
    weight: float = 0.5
    _patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: list(_INJECTION_PATTERNS), repr=False
    )

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Scan *prompt* against compiled injection patterns.

        Returns a high-confidence positive on the first match,
        or a benign result when no patterns match.
        """
        for pattern in self._patterns:
            match = pattern.search(prompt)
            if match:
                return ClassifierResult(
                    score=0.90,
                    label="injection",
                    confidence=0.85,
                    metadata={
                        "matched_pattern": pattern.pattern,
                        "matched_text": match.group(),
                    },
                )

        return ClassifierResult(
            score=0.0,
            label="benign",
            confidence=0.5,
        )
