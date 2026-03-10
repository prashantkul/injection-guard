"""Stage 3: Structural analysis for injection boundary detection."""
from __future__ import annotations

import re

from injection_guard.types import StructuralSignals

__all__ = ["StructuralAnalyzer"]

# Chat / prompt delimiters used by various LLM formats.
_CHAT_DELIMITERS: list[tuple[str, re.Pattern[str]]] = [
    ("<|im_start|>", re.compile(re.escape("<|im_start|>"))),
    ("<|im_end|>", re.compile(re.escape("<|im_end|>"))),
    ("[INST]", re.compile(re.escape("[INST]"))),
    ("<<SYS>>", re.compile(re.escape("<<SYS>>"))),
    ("<s>", re.compile(re.escape("<s>"))),
    ("</s>", re.compile(re.escape("</s>"))),
    ("<|user|>", re.compile(re.escape("<|user|>"))),
    ("<|system|>", re.compile(re.escape("<|system|>"))),
    ("<human>", re.compile(re.escape("<human>"))),
    ("<assistant>", re.compile(re.escape("<assistant>"))),
    ("Human:", re.compile(r"\bHuman:")),
    ("Assistant:", re.compile(r"\bAssistant:")),
]

# XML/HTML tag and comment patterns.
_XML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Instruction boundary patterns.
_BOUNDARY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("END OF USER INPUT", re.compile(r"---\s*END\s+OF\s+USER\s+INPUT\s*---", re.IGNORECASE)),
    ("NEW INSTRUCTIONS", re.compile(r"NEW\s+INSTRUCTIONS\s*:", re.IGNORECASE)),
    ("horizontal rule", re.compile(r"^[-]{3,}$|^[=]{3,}$|^[*]{3,}$", re.MULTILINE)),
    ("END OF pattern", re.compile(r"END\s+OF\b", re.IGNORECASE)),
]

# Characters treated as separators for density calculation.
_SEPARATOR_CHARS = set("-=*_~|+#")


class StructuralAnalyzer:
    """Detects structural injection markers such as chat delimiters and boundaries.

    This is Stage 3 of the preprocessor pipeline.
    """

    def analyze(self, text: str) -> StructuralSignals:
        """Scan *text* for structural injection signals.

        Args:
            text: The input string to analyse.

        Returns:
            A ``StructuralSignals`` dataclass.
        """
        chat_delimiters_found: list[str] = []
        xml_html_tags: list[str] = []
        instruction_boundary_patterns: list[str] = []

        # --- chat delimiters ---
        for label, pattern in _CHAT_DELIMITERS:
            if pattern.search(text):
                chat_delimiters_found.append(label)

        # --- XML/HTML tags ---
        for match in _XML_TAG_RE.finditer(text):
            tag = match.group()
            if tag not in xml_html_tags:
                xml_html_tags.append(tag)

        # --- HTML comments ---
        for match in _HTML_COMMENT_RE.finditer(text):
            comment = match.group()
            if comment not in xml_html_tags:
                xml_html_tags.append(comment)

        # --- instruction boundary patterns ---
        for label, pattern in _BOUNDARY_PATTERNS:
            if pattern.search(text):
                instruction_boundary_patterns.append(label)

        # --- separator density ---
        separator_density = self._separator_density(text)

        return StructuralSignals(
            chat_delimiters_found=chat_delimiters_found,
            xml_html_tags=xml_html_tags,
            instruction_boundary_patterns=instruction_boundary_patterns,
            separator_density=separator_density,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _separator_density(text: str) -> float:
        """Compute ratio of separator characters to total characters."""
        if not text:
            return 0.0
        count = sum(1 for ch in text if ch in _SEPARATOR_CHARS)
        return count / len(text)
