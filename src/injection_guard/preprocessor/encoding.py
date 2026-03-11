"""Stage 2: Encoding detection and decoding."""
from __future__ import annotations

import base64
import html
import re
from urllib.parse import unquote

from injection_guard.types import EncodingSignals

__all__ = ["EncodingDetector"]

# Patterns for encoded content.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{4,}={0,2}")
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}){4,}")  # 8+ hex chars, pairs
_URL_ENCODED_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){2,}")
_HTML_ENTITY_RE = re.compile(r"(?:&#\d+;|&#x[0-9a-fA-F]+;|&[a-zA-Z]+;)")


class EncodingDetector:
    """Detects and decodes base64, hex, URL, and HTML entity encodings.

    This is Stage 2 of the preprocessor pipeline.
    """

    def analyze(self, text: str) -> EncodingSignals:
        """Scan *text* for encoded payloads and return signals.

        Args:
            text: The (possibly normalized) input string.

        Returns:
            An ``EncodingSignals`` dataclass with detected encodings.
        """
        encodings_found: list[str] = []
        decoded_payloads: list[str] = []
        total_encoded_chars = 0

        # --- base64 ---
        for match in _BASE64_RE.finditer(text):
            candidate = match.group()
            if len(candidate) % 4 != 0:
                continue
            decoded = self._try_base64(candidate)
            if decoded is not None:
                if "base64" not in encodings_found:
                    encodings_found.append("base64")
                decoded_payloads.append(decoded)
                total_encoded_chars += len(candidate)

        # --- hex ---
        for match in _HEX_RE.finditer(text):
            candidate = match.group()
            decoded = self._try_hex(candidate)
            if decoded is not None:
                if "hex" not in encodings_found:
                    encodings_found.append("hex")
                decoded_payloads.append(decoded)
                total_encoded_chars += len(candidate)

        # --- URL encoding ---
        for match in _URL_ENCODED_RE.finditer(text):
            candidate = match.group()
            decoded = unquote(candidate)
            if decoded != candidate:
                if "url" not in encodings_found:
                    encodings_found.append("url")
                decoded_payloads.append(decoded)
                total_encoded_chars += len(candidate)

        # --- HTML entities ---
        entity_matches = _HTML_ENTITY_RE.findall(text)
        if entity_matches:
            raw_entities = "".join(entity_matches)
            decoded = html.unescape(raw_entities)
            if decoded != raw_entities:
                encodings_found.append("html_entity")
                decoded_payloads.append(decoded)
                total_encoded_chars += sum(len(m) for m in entity_matches)

        # --- encoding density ---
        encoding_density = total_encoded_chars / len(text) if text else 0.0

        # --- nested encoding check ---
        nested_encoding = self._check_nested(decoded_payloads)

        return EncodingSignals(
            encodings_found=encodings_found,
            decoded_payloads=decoded_payloads,
            encoding_density=encoding_density,
            nested_encoding=nested_encoding,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_base64(candidate: str) -> str | None:
        """Try to decode *candidate* as base64.  Return text or None."""
        try:
            raw = base64.b64decode(candidate, validate=True)
            decoded = raw.decode("utf-8")
            # Heuristic: at least half printable to avoid false positives.
            printable = sum(1 for c in decoded if c.isprintable() or c.isspace())
            if printable >= len(decoded) * 0.5:
                return decoded
        except Exception:
            pass
        return None

    @staticmethod
    def _try_hex(candidate: str) -> str | None:
        """Try to decode *candidate* as hex-encoded UTF-8."""
        try:
            raw = bytes.fromhex(candidate)
            decoded = raw.decode("utf-8")
            printable = sum(1 for c in decoded if c.isprintable() or c.isspace())
            if printable >= len(decoded) * 0.5:
                return decoded
        except Exception:
            pass
        return None

    @staticmethod
    def _check_nested(payloads: list[str]) -> bool:
        """Return True if any decoded payload itself contains encoded content."""
        for payload in payloads:
            if _BASE64_RE.search(payload) and len(payload) >= 8:
                candidate = _BASE64_RE.search(payload)
                if candidate and len(candidate.group()) % 4 == 0:
                    return True
            if _URL_ENCODED_RE.search(payload):
                return True
            if _HEX_RE.search(payload):
                return True
            if _HTML_ENTITY_RE.search(payload):
                return True
        return False
