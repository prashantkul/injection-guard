"""Majority-voting aggregation strategy for classifier results."""
from __future__ import annotations

from injection_guard.types import BaseClassifier, ClassifierResult

__all__ = ["MajorityVotingAggregator"]


class MajorityVotingAggregator:
    """Combines classifier outputs using majority voting.

    Each classifier casts a binary vote based on its label.  The ensemble
    label is determined by whichever class receives more votes, and the
    ensemble score is the proportion of ``"injection"`` votes.
    """

    def aggregate(
        self, results: list[tuple[BaseClassifier, ClassifierResult]]
    ) -> tuple[float, str]:
        """Compute the majority-vote ensemble score and label.

        Args:
            results: Pairs of ``(classifier, result)`` produced by the
                router.

        Returns:
            A ``(score, label)`` tuple where *score* is the fraction of
            injection votes and *label* is the majority class.
        """
        if not results:
            return 0.0, "benign"

        total = len(results)
        injection_votes = sum(
            1 for _, r in results if r.label == "injection"
        )

        score = injection_votes / total
        label = "injection" if injection_votes > total / 2 else "benign"
        return score, label
