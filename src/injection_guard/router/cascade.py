"""Cascade router — runs classifiers tier-by-tier, fast → medium → slow.

Exits early when a single classifier produces a confident result. Supports
retry with exponential backoff and optional fast-tier skipping when the
preprocessor risk prior is high.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from injection_guard.types import (
    BaseClassifier,
    CascadeConfig,
    ClassifierResult,
    RouteResult,
    SignalVector,
)

__all__ = ["CascadeRouter"]

logger = logging.getLogger(__name__)

_TIER_ORDER: list[Literal["fast", "medium", "slow"]] = ["fast", "medium", "slow"]


class CascadeRouter:
    """Routes prompts through classifiers in latency-tier order.

    Classifiers are grouped by their ``latency_tier`` attribute and executed
    from fastest to slowest.  If a classifier returns a score that exceeds the
    configured ``fast_confidence`` threshold (i.e. confidently benign *or*
    confidently injection), routing stops immediately.

    Args:
        config: A ``CascadeConfig`` instance controlling timeouts, retries,
            confidence thresholds, and risk-prior escalation behaviour.
    """

    def __init__(self, config: CascadeConfig) -> None:
        self._config = config

    async def route(
        self,
        classifiers: list[BaseClassifier],
        prompt: str,
        signals: SignalVector | None = None,
        risk_prior: float = 0.0,
    ) -> RouteResult:
        """Run classifiers in cascade order and return results.

        Args:
            classifiers: Classifiers to invoke, in any order.
            prompt: The prompt text to classify.
            signals: Optional preprocessor signal vector.
            risk_prior: Risk prior from the preprocessor (0.0–1.0).

        Returns:
            A ``RouteResult`` containing per-classifier results and whether
            at least one classifier responded successfully.
        """
        results: list[tuple[str, ClassifierResult]] = []
        total_classifiers = 0
        failed_classifiers = 0

        # Group classifiers by tier.
        tiers: dict[str, list[BaseClassifier]] = {t: [] for t in _TIER_ORDER}
        for clf in classifiers:
            tier = clf.latency_tier
            if tier in tiers:
                tiers[tier].append(clf)

        skip_fast = (
            self._config.escalate_on_high_risk_prior
            and risk_prior > self._config.risk_prior_escalation_threshold
        )

        for tier in _TIER_ORDER:
            if skip_fast and tier == "fast":
                logger.debug(
                    "Skipping fast tier (risk_prior=%.3f > %.3f)",
                    risk_prior,
                    self._config.risk_prior_escalation_threshold,
                )
                continue

            for clf in tiers[tier]:
                total_classifiers += 1
                result = await self._invoke_with_retry(clf, prompt, signals)
                if result is None:
                    failed_classifiers += 1
                    continue

                results.append((clf.name, result))

                # Early exit on confident result.
                if self._is_confident(result.score):
                    return RouteResult(results=results, quorum_met=True)

        # Quorum met if at least half of attempted classifiers responded
        quorum_met = len(results) > 0 and len(results) >= (total_classifiers + 1) // 2
        return RouteResult(results=results, quorum_met=quorum_met)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_confident(self, score: float) -> bool:
        """Return ``True`` if the score is confidently benign or injection.

        Args:
            score: The classifier score in [0.0, 1.0].

        Returns:
            Whether the score exceeds the fast-confidence threshold in either
            direction.
        """
        threshold = self._config.fast_confidence
        return score > threshold or score < (1.0 - threshold)

    async def _invoke_with_retry(
        self,
        clf: BaseClassifier,
        prompt: str,
        signals: SignalVector | None,
    ) -> ClassifierResult | None:
        """Invoke a classifier with timeout and exponential-backoff retry.

        Args:
            clf: The classifier to invoke.
            prompt: The prompt text.
            signals: Optional signal vector.

        Returns:
            The classifier result, or ``None`` if all attempts failed.
        """
        timeout_s = self._config.timeout_ms / 1000.0

        for attempt in range(self._config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    clf.classify(prompt, signals),
                    timeout=timeout_s,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Classifier %s timed out (attempt %d/%d)",
                    clf.name,
                    attempt + 1,
                    self._config.max_retries + 1,
                )
            except Exception:
                logger.warning(
                    "Classifier %s raised an error (attempt %d/%d)",
                    clf.name,
                    attempt + 1,
                    self._config.max_retries + 1,
                    exc_info=True,
                )

            # Backoff before retry (skip if this was the last attempt).
            if attempt < self._config.max_retries:
                backoff_s = (
                    self._config.backoff_base_ms * (2**attempt)
                ) / 1000.0
                await asyncio.sleep(backoff_s)

        logger.error(
            "Classifier %s failed after %d attempts — skipping",
            clf.name,
            self._config.max_retries + 1,
        )
        return None
