"""Stage 5: GLiNER-based semantic entity detection (optional dependency)."""
from __future__ import annotations

from injection_guard.types import EntityMatch, LinguisticSignals

__all__ = ["GLiNERAnalyzer"]

_DEFAULT_LABELS: list[str] = [
    "instruction override",
    "role assignment",
    "system prompt reference",
    "safety bypass",
    "mode switch",
    "authority claim",
    "output format manipulation",
    "context boundary",
    "prompt leaking request",
    "jailbreak technique",
]

try:
    from gliner import GLiNER as _GLiNER  # type: ignore[import-untyped]

    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False


class GLiNERAnalyzer:
    """Wraps GLiNER for injection-oriented named entity recognition.

    This is Stage 5 of the preprocessor pipeline.  GLiNER is an optional
    dependency; when it is not installed the analyzer returns empty signals.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_base",
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialise the analyzer.

        Args:
            model_name: HuggingFace model identifier for GLiNER.
            labels: Entity labels to detect.  Defaults to the built-in
                injection-oriented label set.
            threshold: Minimum confidence score for entity predictions.
        """
        self._labels = labels or list(_DEFAULT_LABELS)
        self._threshold = threshold
        self._model: object | None = None
        self._model_name = model_name

    @property
    def available(self) -> bool:
        """Return True if GLiNER is installed."""
        return _GLINER_AVAILABLE

    def _load_model(self) -> None:
        """Lazily load the GLiNER model on first use."""
        if not _GLINER_AVAILABLE:
            return
        if self._model is None:
            self._model = _GLiNER.from_pretrained(self._model_name)

    def analyze(self, text: str) -> LinguisticSignals:
        """Run GLiNER entity detection on *text*.

        Args:
            text: The input string.

        Returns:
            A ``LinguisticSignals`` dataclass.  If GLiNER is not installed,
            all fields are empty / zero.
        """
        if not _GLINER_AVAILABLE:
            return LinguisticSignals()

        try:
            self._load_model()
        except Exception:
            return LinguisticSignals()

        try:
            raw_entities = self._model.predict_entities(  # type: ignore[union-attr]
                text, self._labels, threshold=self._threshold
            )
        except Exception:
            return LinguisticSignals()

        entities: list[EntityMatch] = []
        types_found: list[str] = []
        max_confidence = 0.0

        for ent in raw_entities:
            match = EntityMatch(
                text=ent.get("text", ""),
                label=ent.get("label", ""),
                score=float(ent.get("score", 0.0)),
            )
            entities.append(match)

            if match.label and match.label not in types_found:
                types_found.append(match.label)

            if match.score > max_confidence:
                max_confidence = match.score

        return LinguisticSignals(
            injection_entities=entities,
            entity_types_found=types_found,
            max_entity_confidence=max_confidence,
            entity_count=len(entities),
        )
