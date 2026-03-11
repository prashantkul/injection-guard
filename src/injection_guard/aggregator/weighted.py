"""Weighted-average aggregation strategy for classifier results."""
from __future__ import annotations

from injection_guard.types import BaseClassifier, ClassifierResult

__all__ = ["WeightedAverageAggregator"]


class WeightedAverageAggregator:
    """Combines classifier scores using a weighted average.

    Each classifier contributes its score proportionally to its configured
    weight.  The ensemble score is the weighted mean across all classifiers,
    and the label is ``"injection"`` when the score reaches or exceeds 0.5.
    """

    def aggregate(
        self, results: list[tuple[BaseClassifier, ClassifierResult]]
    ) -> tuple[float, str]:
        """Compute the weighted-average ensemble score and label.

        Args:
            results: Pairs of ``(classifier, result)`` produced by the
                router.

        Returns:
            A ``(score, label)`` tuple where *score* is in ``[0, 1]`` and
            *label* is either ``"injection"`` or ``"benign"``.
        """
        if not results:
            return 0.0, "benign"

        weighted_sum = sum(r.score * c.weight for c, r in results)
        weight_total = sum(c.weight for c, _ in results)

        if weight_total == 0:
            return 0.0, "benign"

        score = weighted_sum / weight_total
        label = "injection" if score >= 0.5 else "benign"
        return score, label
