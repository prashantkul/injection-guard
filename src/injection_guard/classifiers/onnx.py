"""ONNX Runtime classifier for local model inference."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from injection_guard.types import ClassifierResult, SignalVector

__all__ = ["OnnxClassifier"]

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False


@dataclass
class OnnxClassifier:
    """Classifier backed by an ONNX Runtime inference session.

    Attributes:
        name: Classifier identifier.
        model_path: Filesystem path to the ``.onnx`` model file.
        latency_tier: Always ``"fast"`` for this classifier.
        weight: Weight used by the ensemble aggregator.
    """

    name: str
    model_path: str
    latency_tier: str = "fast"
    weight: float = 1.0
    _session: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not _ORT_AVAILABLE:
            return
        try:
            self._session = ort.InferenceSession(self.model_path)
        except Exception:  # noqa: BLE001
            self._session = None

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Run inference on *prompt* and return a :class:`ClassifierResult`.

        If ``onnxruntime`` is not installed or the session failed to load,
        returns a degraded result with ``confidence=0.0``.
        """
        if not _ORT_AVAILABLE:
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": "onnxruntime is not installed"},
            )

        if self._session is None:
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": "ONNX session failed to initialise"},
            )

        try:
            start = time.perf_counter()
            input_name = self._session.get_inputs()[0].name

            try:
                import numpy as np  # type: ignore[import-untyped]
                input_feed = {input_name: np.array([[prompt]], dtype=np.object_)}
            except ImportError:
                input_feed = {input_name: [[prompt]]}

            outputs = self._session.run(None, input_feed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Expect first output to be probabilities [benign, injection]
            probs = outputs[0]
            if hasattr(probs, "tolist"):
                probs = probs.tolist()
            # Flatten if nested
            if isinstance(probs[0], list):
                probs = probs[0]

            injection_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
            label = "injection" if injection_score >= 0.5 else "benign"

            return ClassifierResult(
                score=injection_score,
                label=label,
                confidence=abs(injection_score - 0.5) * 2,
                latency_ms=elapsed_ms,
                metadata={"raw_probs": probs},
            )
        except Exception as exc:  # noqa: BLE001
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": str(exc)},
            )
