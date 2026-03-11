"""Tests for the MajorityVotingAggregator."""
from __future__ import annotations

import pytest

from injection_guard.aggregator.voting import MajorityVotingAggregator
from injection_guard.types import ClassifierResult

from tests.conftest import MockClassifier


@pytest.fixture
def aggregator() -> MajorityVotingAggregator:
    return MajorityVotingAggregator()


class TestClearMajority:
    """Test voting with a clear majority."""

    def test_injection_majority(self, aggregator: MajorityVotingAggregator):
        clfs = [MockClassifier(name=f"c{i}", weight=1.0) for i in range(3)]
        results = [
            (clfs[0], ClassifierResult(score=0.9, label="injection")),
            (clfs[1], ClassifierResult(score=0.8, label="injection")),
            (clfs[2], ClassifierResult(score=0.1, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        assert abs(score - 2 / 3) < 0.01
        assert label == "injection"

    def test_benign_majority(self, aggregator: MajorityVotingAggregator):
        clfs = [MockClassifier(name=f"c{i}", weight=1.0) for i in range(3)]
        results = [
            (clfs[0], ClassifierResult(score=0.1, label="benign")),
            (clfs[1], ClassifierResult(score=0.2, label="benign")),
            (clfs[2], ClassifierResult(score=0.9, label="injection")),
        ]
        score, label = aggregator.aggregate(results)
        assert abs(score - 1 / 3) < 0.01
        assert label == "benign"

    def test_unanimous_injection(self, aggregator: MajorityVotingAggregator):
        clfs = [MockClassifier(name=f"c{i}", weight=1.0) for i in range(3)]
        results = [
            (clfs[0], ClassifierResult(score=0.9, label="injection")),
            (clfs[1], ClassifierResult(score=0.8, label="injection")),
            (clfs[2], ClassifierResult(score=0.7, label="injection")),
        ]
        score, label = aggregator.aggregate(results)
        assert score == 1.0
        assert label == "injection"


class TestTiedVote:
    """Test voting with a tie."""

    def test_even_split_favors_benign(self, aggregator: MajorityVotingAggregator):
        clfs = [MockClassifier(name=f"c{i}", weight=1.0) for i in range(2)]
        results = [
            (clfs[0], ClassifierResult(score=0.9, label="injection")),
            (clfs[1], ClassifierResult(score=0.1, label="benign")),
        ]
        score, label = aggregator.aggregate(results)
        # 1 injection vote out of 2 -> 0.5, not > total/2 -> benign
        assert score == 0.5
        assert label == "benign"


class TestEmptyResults:
    """Test voting with no results."""

    def test_empty_list(self, aggregator: MajorityVotingAggregator):
        score, label = aggregator.aggregate([])
        assert score == 0.0
        assert label == "benign"
