"""Threshold-based decision engine for the injection-guard pipeline."""
from __future__ import annotations

from injection_guard.types import Action, ThresholdConfig

__all__ = ["ThresholdEngine"]


class ThresholdEngine:
    """Maps ensemble scores to allow / flag / block actions.

    The engine applies a simple two-threshold scheme:

    * **score >= block_threshold** --> :attr:`Action.BLOCK`
    * **score >= flag_threshold** --> :attr:`Action.FLAG`
    * **otherwise** --> :attr:`Action.ALLOW`

    An optional *preprocessor_block_threshold* allows the preprocessor's
    risk prior to short-circuit straight to a block before classifiers run.

    Attributes:
        config: The current :class:`ThresholdConfig` governing decisions.
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config

    def decide(self, ensemble_score: float) -> Action:
        """Return the action for a given ensemble score.

        Args:
            ensemble_score: Combined score from the aggregator, in ``[0, 1]``.

        Returns:
            The appropriate :class:`Action` based on configured thresholds.
        """
        if ensemble_score >= self.config.block_threshold:
            return Action.BLOCK
        if ensemble_score >= self.config.flag_threshold:
            return Action.FLAG
        return Action.ALLOW

    def preprocessor_blocks(self, risk_prior: float) -> bool:
        """Check whether the preprocessor risk prior alone warrants a block.

        Args:
            risk_prior: The risk score produced by the preprocessor pipeline.

        Returns:
            ``True`` if the *preprocessor_block_threshold* is configured and
            the *risk_prior* meets or exceeds it, ``False`` otherwise.
        """
        if self.config.preprocessor_block_threshold is None:
            return False
        return risk_prior >= self.config.preprocessor_block_threshold

    def update_thresholds(
        self,
        block: float | None = None,
        flag: float | None = None,
        preprocessor_block: float | None = None,
    ) -> None:
        """Update one or more thresholds at runtime.

        Args:
            block: New block threshold, or ``None`` to keep the current value.
            flag: New flag threshold, or ``None`` to keep the current value.
            preprocessor_block: New preprocessor block threshold.  Pass
                ``None`` to keep the current value (use the config directly
                to disable it by setting it to ``None``).
        """
        if block is not None:
            self.config.block_threshold = block
        if flag is not None:
            self.config.flag_threshold = flag
        if preprocessor_block is not None:
            self.config.preprocessor_block_threshold = preprocessor_block
