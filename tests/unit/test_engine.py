"""Tests for the ThresholdEngine."""
from __future__ import annotations

import pytest

from injection_guard.engine import ThresholdEngine
from injection_guard.types import Action, ThresholdConfig


@pytest.fixture
def engine() -> ThresholdEngine:
    return ThresholdEngine(ThresholdConfig(block_threshold=0.85, flag_threshold=0.50))


class TestDecide:
    """Test the decide method with various scores."""

    def test_score_above_block_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.90) == Action.BLOCK

    def test_score_between_flag_and_block(self, engine: ThresholdEngine):
        assert engine.decide(0.60) == Action.FLAG

    def test_score_below_flag_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.30) == Action.ALLOW

    def test_score_zero(self, engine: ThresholdEngine):
        assert engine.decide(0.0) == Action.ALLOW

    def test_score_one(self, engine: ThresholdEngine):
        assert engine.decide(1.0) == Action.BLOCK


class TestBoundaryValues:
    """Test exact boundary values."""

    def test_exact_block_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.85) == Action.BLOCK

    def test_just_below_block_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.849) == Action.FLAG

    def test_exact_flag_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.50) == Action.FLAG

    def test_just_below_flag_threshold(self, engine: ThresholdEngine):
        assert engine.decide(0.499) == Action.ALLOW


class TestPreprocessorBlockThreshold:
    """Test preprocessor block threshold behaviour."""

    def test_no_preprocessor_threshold_never_blocks(self, engine: ThresholdEngine):
        assert engine.preprocessor_blocks(0.99) is False

    def test_preprocessor_threshold_blocks_above(self):
        engine = ThresholdEngine(
            ThresholdConfig(
                block_threshold=0.85,
                flag_threshold=0.50,
                preprocessor_block_threshold=0.80,
            )
        )
        assert engine.preprocessor_blocks(0.85) is True

    def test_preprocessor_threshold_allows_below(self):
        engine = ThresholdEngine(
            ThresholdConfig(
                block_threshold=0.85,
                flag_threshold=0.50,
                preprocessor_block_threshold=0.80,
            )
        )
        assert engine.preprocessor_blocks(0.70) is False

    def test_preprocessor_exact_threshold(self):
        engine = ThresholdEngine(
            ThresholdConfig(
                block_threshold=0.85,
                flag_threshold=0.50,
                preprocessor_block_threshold=0.80,
            )
        )
        assert engine.preprocessor_blocks(0.80) is True


class TestUpdateThresholds:
    """Test runtime threshold updates."""

    def test_update_block_threshold(self, engine: ThresholdEngine):
        engine.update_thresholds(block=0.90)
        assert engine.config.block_threshold == 0.90
        # Flag should be unchanged
        assert engine.config.flag_threshold == 0.50

    def test_update_flag_threshold(self, engine: ThresholdEngine):
        engine.update_thresholds(flag=0.40)
        assert engine.config.flag_threshold == 0.40

    def test_update_preprocessor_block(self, engine: ThresholdEngine):
        engine.update_thresholds(preprocessor_block=0.75)
        assert engine.config.preprocessor_block_threshold == 0.75

    def test_update_multiple(self, engine: ThresholdEngine):
        engine.update_thresholds(block=0.90, flag=0.40, preprocessor_block=0.70)
        assert engine.config.block_threshold == 0.90
        assert engine.config.flag_threshold == 0.40
        assert engine.config.preprocessor_block_threshold == 0.70

    def test_none_values_keep_current(self, engine: ThresholdEngine):
        engine.update_thresholds(block=None, flag=None)
        assert engine.config.block_threshold == 0.85
        assert engine.config.flag_threshold == 0.50
