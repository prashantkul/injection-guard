"""Parallel router — fires all classifiers concurrently, returns on quorum.

All classifiers are launched as concurrent asyncio tasks.  As soon as enough
classifiers agree on the same label (the *quorum*), remaining tasks are
cancelled and results are returned.  Individual classifier failures are
handled gracefully and never propagate.
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
    they arrive and checks whether ``config.quorum`` classifiers agree on the
    same label.  Once quorum is reached the remaining in-flight tasks are
    cancelled.

    Args:
        config: A ``ParallelConfig`` instance controlling the timeout, retry
            policy, and quorum size.
    """

    def __init__(self, config: ParallelConfig) -> None:
        self._config = config

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
        quorum_event = asyncio.Event()
        timeout_s = self._config.timeout_ms / 1000.0

        async def _run(clf: BaseClassifier) -> None:
            """Execute a single classifier and record its result.

            Args:
                clf: The classifier to run.
            """
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
