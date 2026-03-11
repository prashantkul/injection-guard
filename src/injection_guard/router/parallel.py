"""Parallel router — fires all classifiers concurrently, returns on quorum.

All classifiers are launched as concurrent asyncio tasks.  Supports two
quorum modes:

1. **Simple quorum**: Wait for N classifiers to agree on the same label.
2. **Category quorum**: Wait for at least N responses per category
   (e.g. 1 from "local", 1 from "api").

Individual classifier failures are handled gracefully and never propagate.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

from injection_guard.types import (
    BaseClassifier,
    ClassifierResult,
    ParallelConfig,
    SignalVector,
)

__all__ = ["ParallelRouter"]

logger = logging.getLogger(__name__)


class ParallelRouter:
    """Routes prompts to all classifiers in parallel, returning on quorum.

    Every classifier is invoked concurrently.  The router collects results as
    they arrive and checks the quorum condition.  Once satisfied, remaining
    in-flight tasks are cancelled.

    Args:
        config: A ``ParallelConfig`` instance controlling the timeout, retry
            policy, and quorum size / category quorum.
    """

    def __init__(self, config: ParallelConfig) -> None:
        self._config = config
        self._use_categories = bool(config.category_quorum)

    async def route(
        self,
        classifiers: list[BaseClassifier],
        prompt: str,
        signals: SignalVector | None = None,
        risk_prior: float = 0.0,
    ) -> list[tuple[str, ClassifierResult]]:
        """Run all classifiers concurrently and return on quorum or timeout.

        Args:
            classifiers: Classifiers to invoke.
            prompt: The prompt text to classify.
            signals: Optional preprocessor signal vector.
            risk_prior: Risk prior from the preprocessor (unused by this
                router but accepted for interface compatibility).

        Returns:
            A list of ``(classifier_name, ClassifierResult)`` tuples for every
            classifier that completed before the quorum was met or the global
            timeout expired.
        """
        if not classifiers:
            return []

        results: list[tuple[str, ClassifierResult]] = []
        label_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        quorum_event = asyncio.Event()
        timeout_s = self._config.timeout_ms / 1000.0
        cat_map = self._config.classifier_categories
        cat_quorum = self._config.category_quorum

        async def _run(clf: BaseClassifier) -> None:
            """Execute a single classifier and record its result."""
            try:
                result = await clf.classify(prompt, signals)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Classifier %s failed", clf.name, exc_info=True
                )
                return

            results.append((clf.name, result))
            label_counts[result.label] += 1

            # Track category completion
            category = cat_map.get(clf.name, "")
            if category:
                category_counts[category] += 1

            # Check quorum
            if self._use_categories:
                if self._category_quorum_met(category_counts, cat_quorum):
                    quorum_event.set()
            else:
                if label_counts[result.label] >= self._config.quorum:
                    quorum_event.set()

        tasks = [asyncio.create_task(_run(clf)) for clf in classifiers]

        try:
            await asyncio.wait_for(quorum_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "Parallel router timed out after %.0f ms with %d/%d results",
                self._config.timeout_ms,
                len(results),
                len(classifiers),
            )
        finally:
            # Cancel any tasks still running.
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellation to propagate so we don't leak tasks.
            await asyncio.gather(*tasks, return_exceptions=True)

        return results

    @staticmethod
    def _category_quorum_met(
        counts: Counter[str], required: dict[str, int]
    ) -> bool:
        """Check if all category quorum requirements are satisfied."""
        return all(
            counts.get(cat, 0) >= min_count
            for cat, min_count in required.items()
        )
