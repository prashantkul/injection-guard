"""Preprocessor pipeline that orchestrates all five analysis stages."""
from __future__ import annotations

from injection_guard.types import (
    PreprocessorOutput,
    SignalVector,
)

from injection_guard.preprocessor.unicode import UnicodeNormalizer
from injection_guard.preprocessor.encoding import EncodingDetector
from injection_guard.preprocessor.structural import StructuralAnalyzer
from injection_guard.preprocessor.token import TokenBoundaryDetector
from injection_guard.preprocessor.gliner import GLiNERAnalyzer

__all__ = ["Preprocessor"]


class Preprocessor:
    """Runs the five-stage preprocessor pipeline and computes a risk prior.

    Stages:
        1. Unicode normalization (always)
        2. Encoding detection (always)
        3. Structural analysis (always)
        4. Token boundary detection (always)
        5. GLiNER semantic analysis (only if the library is installed)
    """

    def __init__(self, *, gliner_model: str = "urchade/gliner_base") -> None:
        """Initialise the preprocessor with its sub-analyzers.

        Args:
            gliner_model: HuggingFace model name for the GLiNER stage.
        """
        self._unicode = UnicodeNormalizer()
        self._encoding = EncodingDetector()
        self._structural = StructuralAnalyzer()
        self._token = TokenBoundaryDetector()
        self._gliner = GLiNERAnalyzer(model_name=gliner_model)

    def process(self, prompt: str) -> PreprocessorOutput:
        """Run all preprocessing stages on *prompt*.

        Args:
            prompt: The raw user prompt.

        Returns:
            A ``PreprocessorOutput`` containing the normalized prompt,
            all signal vectors, decoded payloads, and the computed risk prior.
        """
        # Stage 1: Unicode normalization
        normalized, unicode_signals = self._unicode.analyze(prompt)

        # Stage 2: Encoding detection (on normalized text)
        encoding_signals = self._encoding.analyze(normalized)

        # Stage 3: Structural analysis
        structural_signals = self._structural.analyze(normalized)

        # Stage 4: Token boundary detection
        token_signals = self._token.analyze(normalized)

        # Stage 5: GLiNER (optional)
        linguistic_signals = self._gliner.analyze(normalized)

        signals = SignalVector(
            unicode=unicode_signals,
            encoding=encoding_signals,
            structural=structural_signals,
            token=token_signals,
            linguistic=linguistic_signals,
        )

        risk_prior = self._compute_risk_prior(signals)

        return PreprocessorOutput(
            normalized_prompt=normalized,
            original_prompt=prompt,
            decoded_payloads=encoding_signals.decoded_payloads,
            signals=signals,
            risk_prior=risk_prior,
        )

    @staticmethod
    def _compute_risk_prior(signals: SignalVector) -> float:
        """Compute a heuristic risk prior from signal weights (Appendix A).

        Args:
            signals: The combined signal vector from all stages.

        Returns:
            A float in [0.0, 1.0].
        """
        score = 0.0

        # Unicode signals
        if signals.unicode.homoglyph_count > 3:
            score += 0.3
        if signals.unicode.zero_width_count > 0:
            score += 0.2
        if signals.unicode.bidi_override_count > 0:
            score += 0.3
        if signals.unicode.script_mixing:
            score += 0.2
        if signals.unicode.normalization_edit_distance > 10:
            score += 0.2

        # Encoding signals
        if signals.encoding.nested_encoding:
            score += 0.4
        if signals.encoding.encoding_density > 0.3:
            score += 0.2

        # Structural signals
        if signals.structural.chat_delimiters_found:
            score += 0.5

        # Linguistic signals
        if signals.linguistic.entity_count >= 2:
            score += 0.5
        elif signals.linguistic.entity_count == 1:
            score += 0.3
        if signals.linguistic.max_entity_confidence > 0.8:
            score += 0.2
        if (
            "instruction override" in signals.linguistic.entity_types_found
            and "role assignment" in signals.linguistic.entity_types_found
        ):
            score += 0.3

        return min(1.0, score)
