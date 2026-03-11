"""Tests for the MetaClassifierAggregator."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from injection_guard.aggregator.meta import MetaClassifierAggregator
from injection_guard.types import ClassifierResult

from tests.conftest import MockClassifier


class TestFallbackToWeighted:
    """Test that the meta-classifier falls back to weighted average when no model."""

    def test_missing_model_file_uses_fallback(self, tmp_path):
        model_path = tmp_path / "nonexistent_model.pkl"
        aggregator = MetaClassifierAggregator(model_path=str(model_path))

        clf = MockClassifier(name="test", weight=1.0)
        results = [(clf, ClassifierResult(score=0.8, label="injection"))]
        score, label = aggregator.aggregate(results)
        assert score == 0.8
        assert label == "injection"

    def test_empty_results_returns_benign(self, tmp_path):
        model_path = tmp_path / "nonexistent_model.pkl"
        aggregator = MetaClassifierAggregator(model_path=str(model_path))
        score, label = aggregator.aggregate([])
        assert score == 0.0
        assert label == "benign"

    def test_fallback_weighted_average(self, tmp_path):
        model_path = tmp_path / "nonexistent_model.pkl"
        aggregator = MetaClassifierAggregator(model_path=str(model_path))

        clf_a = MockClassifier(name="a", weight=1.0)
        clf_b = MockClassifier(name="b", weight=1.0)
        results = [
            (clf_a, ClassifierResult(score=0.9, label="injection")),
            (clf_b, ClassifierResult(score=0.1, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        assert abs(score - 0.5) < 0.01
