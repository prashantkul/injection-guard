"""Aggregator module for combining classifier results into ensemble decisions."""
from __future__ import annotations

from injection_guard.aggregator.meta import MetaClassifierAggregator
from injection_guard.aggregator.voting import MajorityVotingAggregator
from injection_guard.aggregator.weighted import WeightedAverageAggregator
from injection_guard.types import AggregatorType

__all__ = [
    "get_aggregator",
    "WeightedAverageAggregator",
    "MajorityVotingAggregator",
    "MetaClassifierAggregator",
]


def get_aggregator(
    aggregator_type: AggregatorType | str,
    *,
    meta_model_path: str | None = None,
) -> WeightedAverageAggregator | MajorityVotingAggregator | MetaClassifierAggregator:
    """Factory that returns an aggregator instance for the given strategy.

    Args:
        aggregator_type: The aggregation strategy to use.  Accepts either an
            :class:`~injection_guard.types.AggregatorType` enum member or its
            string value (e.g. ``"weighted_average"``).
        meta_model_path: Filesystem path to a trained meta-classifier model.
            Only used when *aggregator_type* is ``META_CLASSIFIER``.

    Returns:
        An aggregator instance matching the requested strategy.

    Raises:
        ValueError: If the aggregator type is not recognised.
    """
    if isinstance(aggregator_type, str):
        aggregator_type = AggregatorType(aggregator_type)

    if aggregator_type == AggregatorType.WEIGHTED_AVERAGE:
        return WeightedAverageAggregator()

    if aggregator_type == AggregatorType.VOTING:
        return MajorityVotingAggregator()

    if aggregator_type == AggregatorType.META_CLASSIFIER:
        if meta_model_path is None:
            raise ValueError(
                "meta_model_path is required for the META_CLASSIFIER aggregator."
            )
        return MetaClassifierAggregator(meta_model_path)

    raise ValueError(f"Unknown aggregator type: {aggregator_type!r}")
