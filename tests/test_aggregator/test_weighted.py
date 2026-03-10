"""Tests for the WeightedAverageAggregator."""
from __future__ import annotations

import pytest

from injection_guard.aggregator.weighted import WeightedAverageAggregator
from injection_guard.types import ClassifierResult

from tests.conftest import MockClassifier


@pytest.fixture
def aggregator() -> WeightedAverageAggregator:
    return WeightedAverageAggregator()


class TestEqualWeights:
    """Test aggregation with equal weights."""

    def test_two_classifiers_equal_weight(self, aggregator: WeightedAverageAggregator):
        clf_a = MockClassifier(name="a", weight=1.0)
        clf_b = MockClassifier(name="b", weight=1.0)
        results = [
            (clf_a, ClassifierResult(score=0.8, label="injection")),
            (clf_b, ClassifierResult(score=0.2, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        assert abs(score - 0.5) < 0.01
        assert label == "injection"  # 0.5 >= 0.5

    def test_three_classifiers_equal_weight(self, aggregator: WeightedAverageAggregator):
        clfs = [MockClassifier(name=f"c{i}", weight=1.0) for i in range(3)]
        results = [
            (clfs[0], ClassifierResult(score=0.9, label="injection")),
            (clfs[1], ClassifierResult(score=0.1, label="benign")),
            (clfs[2], ClassifierResult(score=0.3, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        # (0.9 + 0.1 + 0.3) / 3 = 0.4333
        assert abs(score - (0.9 + 0.1 + 0.3) / 3) < 0.01
        assert label == "benign"


class TestUnequalWeights:
    """Test aggregation with different weights."""

    def test_heavy_weight_on_injection(self, aggregator: WeightedAverageAggregator):
        clf_a = MockClassifier(name="a", weight=3.0)
        clf_b = MockClassifier(name="b", weight=1.0)
        results = [
            (clf_a, ClassifierResult(score=0.9, label="injection")),
            (clf_b, ClassifierResult(score=0.1, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        # (0.9*3 + 0.1*1) / (3+1) = 2.8/4 = 0.7
        assert abs(score - 0.7) < 0.01
        assert label == "injection"

    def test_heavy_weight_on_benign(self, aggregator: WeightedAverageAggregator):
        clf_a = MockClassifier(name="a", weight=1.0)
        clf_b = MockClassifier(name="b", weight=3.0)
        results = [
            (clf_a, ClassifierResult(score=0.9, label="injection")),
            (clf_b, ClassifierResult(score=0.1, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        # (0.9*1 + 0.1*3) / (1+3) = 1.2/4 = 0.3
        assert abs(score - 0.3) < 0.01
        assert label == "benign"


class TestSingleClassifier:
    """Test aggregation with a single classifier."""

    def test_single_classifier(self, aggregator: WeightedAverageAggregator):
        clf = MockClassifier(name="only", weight=2.0)
        results = [(clf, ClassifierResult(score=0.85, label="injection"))]
        score, label = aggregator.aggregate(results)
        assert score == 0.85
        assert label == "injection"


class TestEmptyResults:
    """Test aggregation with no results."""

    def test_empty_list(self, aggregator: WeightedAverageAggregator):
        score, label = aggregator.aggregate([])
        assert score == 0.0
        assert label == "benign"
