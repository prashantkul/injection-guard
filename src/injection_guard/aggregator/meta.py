"""Meta-classifier aggregation strategy for classifier results."""
from __future__ import annotations

import logging
from pathlib import Path

from injection_guard.aggregator.weighted import WeightedAverageAggregator
from injection_guard.types import BaseClassifier, ClassifierResult

__all__ = ["MetaClassifierAggregator"]

_logger = logging.getLogger(__name__)


class MetaClassifierAggregator:
    """Aggregates classifier results using a trained meta-classifier.

    A meta-classifier (also known as a *stacking* model) takes the raw
    scores from the individual classifiers as features and produces a single
    ensemble prediction.  The model is loaded from a serialised file on disk
    (joblib or pickle format).

    If the model file cannot be loaded — because the path does not exist,
    the serialisation library is missing, or the file is corrupt — the
    aggregator transparently falls back to a
    :class:`~injection_guard.aggregator.weighted.WeightedAverageAggregator`.

    Attributes:
        model_path: Filesystem path to the serialised meta-classifier.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model: object | None = None
        self._fallback = WeightedAverageAggregator()
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self, results: list[tuple[BaseClassifier, ClassifierResult]]
    ) -> tuple[float, str]:
        """Produce an ensemble score and label via the meta-classifier.

        Args:
            results: Pairs of ``(classifier, result)`` produced by the
                router.

        Returns:
            A ``(score, label)`` tuple.  Falls back to weighted averaging
            when the meta-classifier is unavailable.
        """
        if self._model is None or not results:
            return self._fallback.aggregate(results)

        try:
            import numpy as np  # type: ignore[import-untyped]

            features = np.array(
                [[r.score for _, r in results]]
            )  # shape (1, n_classifiers)

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(features)[0]  # type: ignore[union-attr]
                # Convention: class 1 == "injection"
                score = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                score = float(self._model.predict(features)[0])  # type: ignore[union-attr]

            label = "injection" if score >= 0.5 else "benign"
            return score, label

        except Exception:  # noqa: BLE001
            _logger.warning(
                "Meta-classifier prediction failed; falling back to weighted average.",
                exc_info=True,
            )
            return self._fallback.aggregate(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to deserialise the meta-classifier from disk."""
        if not self.model_path.exists():
            _logger.info(
                "Meta-classifier model not found at %s; "
                "will use weighted-average fallback.",
                self.model_path,
            )
            return

        # Prefer joblib (faster for sklearn models), fall back to pickle.
        try:
            import joblib  # type: ignore[import-untyped]

            self._model = joblib.load(self.model_path)
            _logger.info("Meta-classifier loaded from %s via joblib.", self.model_path)
            return
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            _logger.warning(
                "joblib failed to load %s; trying pickle.", self.model_path,
                exc_info=True,
            )

        try:
            import pickle  # noqa: S403

            with open(self.model_path, "rb") as fh:
                self._model = pickle.load(fh)  # noqa: S301
            _logger.info(
                "Meta-classifier loaded from %s via pickle.", self.model_path
            )
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to load meta-classifier from %s; "
                "will use weighted-average fallback.",
                self.model_path,
                exc_info=True,
            )
