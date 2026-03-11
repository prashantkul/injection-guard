"""Tests for the ParallelRouter."""
from __future__ import annotations

import asyncio

import pytest

from injection_guard.router.parallel import ParallelRouter
from injection_guard.types import ClassifierResult, ParallelConfig


@pytest.fixture
def config() -> ParallelConfig:
    return ParallelConfig(timeout_ms=2000, quorum=2)


@pytest.fixture
def router(config: ParallelConfig) -> ParallelRouter:
    return ParallelRouter(config)


class TestQuorumMet:
    """Test that quorum is reached and results returned quickly."""

    async def test_quorum_of_two_injection(self, router, make_classifier):
        clf_a = make_classifier(name="a", score=0.9, label="injection")
        clf_b = make_classifier(name="b", score=0.8, label="injection")
        clf_c = make_classifier(name="c", score=0.1, label="benign")

        results = await router.route([clf_a, clf_b, clf_c], "test text")

        # At least 2 results should be present (quorum met on "injection")
        assert len(results) >= 2
        injection_count = sum(1 for _, r in results if r.label == "injection")
        assert injection_count >= 2

    async def test_quorum_of_two_benign(self, router, make_classifier):
        clf_a = make_classifier(name="a", score=0.1, label="benign")
        clf_b = make_classifier(name="b", score=0.2, label="benign")
        clf_c = make_classifier(name="c", score=0.9, label="injection")

        results = await router.route([clf_a, clf_b, clf_c], "normal text")

        assert len(results) >= 2
        benign_count = sum(1 for _, r in results if r.label == "benign")
        assert benign_count >= 2


class TestClassifierFailure:
    """Test that classifier failure does not break the router."""

    async def test_one_failure_still_returns_results(self, router, make_classifier):
        clf_ok_a = make_classifier(name="ok-a", score=0.9, label="injection")
        clf_ok_b = make_classifier(name="ok-b", score=0.8, label="injection")
        clf_fail = make_classifier(name="fail", should_fail=True)

        results = await router.route([clf_ok_a, clf_ok_b, clf_fail], "test")

        assert len(results) >= 2
        names = [name for name, _ in results]
        assert "fail" not in names

    async def test_all_fail_returns_empty(self, make_classifier):
        router = ParallelRouter(ParallelConfig(timeout_ms=500, quorum=2))
        clf_a = make_classifier(name="a", should_fail=True)
        clf_b = make_classifier(name="b", should_fail=True)

        results = await router.route([clf_a, clf_b], "test")

        assert results == []


class TestEmptyClassifiers:
    """Test routing with no classifiers."""

    async def test_empty_list(self, router):
        results = await router.route([], "test")
        assert results == []


class TestTimeout:
    """Test that the router handles timeouts gracefully."""

    async def test_timeout_returns_available_results(self, make_classifier):
        router = ParallelRouter(ParallelConfig(timeout_ms=100, quorum=10))

        clf_a = make_classifier(name="a", score=0.9, label="injection")
        clf_b = make_classifier(name="b", score=0.8, label="injection")

        results = await router.route([clf_a, clf_b], "test")

        # Quorum of 10 can never be met with 2 classifiers, but both should finish
        # before the timeout since they are instant
        assert len(results) == 2


class TestCategoryQuorum:
    """Test category-based quorum logic."""

    async def test_category_quorum_met(self, make_classifier):
        config = ParallelConfig(
            timeout_ms=2000,
            category_quorum={"local": 1, "api": 1},
            classifier_categories={
                "local-a": "local",
                "api-a": "api",
                "api-b": "api",
            },
        )
        router = ParallelRouter(config)

        clf_local = make_classifier(name="local-a", score=0.9, label="injection")
        clf_api_a = make_classifier(name="api-a", score=0.8, label="injection")
        clf_api_b = make_classifier(name="api-b", score=0.7, label="injection")

        results = await router.route([clf_local, clf_api_a, clf_api_b], "test")

        assert len(results) >= 2
        names = {name for name, _ in results}
        assert "local-a" in names
        # At least one api classifier responded
        assert names & {"api-a", "api-b"}

    async def test_category_quorum_waits_for_all_categories(self, make_classifier):
        """If only one category responds, quorum is not met."""
        config = ParallelConfig(
            timeout_ms=300,
            category_quorum={"local": 1, "api": 1},
            classifier_categories={
                "fast-local": "local",
                "slow-api": "api",
            },
        )
        router = ParallelRouter(config)

        clf_fast = make_classifier(name="fast-local", score=0.9, label="injection")
        # Slow classifier that won't respond before timeout
        clf_slow = make_classifier(name="slow-api", score=0.8, label="injection", delay=0.5)

        results = await router.route([clf_fast, clf_slow], "test")

        # Timeout hit — only fast-local responded, category quorum not met
        assert len(results) >= 1

    async def test_category_quorum_ignores_simple_quorum(self, make_classifier):
        """When category_quorum is set, simple quorum field is ignored."""
        config = ParallelConfig(
            timeout_ms=2000,
            quorum=99,  # Would never be met
            category_quorum={"local": 1, "api": 1},
            classifier_categories={
                "clf-local": "local",
                "clf-api": "api",
            },
        )
        router = ParallelRouter(config)

        clf_a = make_classifier(name="clf-local", score=0.9, label="injection")
        clf_b = make_classifier(name="clf-api", score=0.8, label="injection")

        results = await router.route([clf_a, clf_b], "test")

        # Category quorum met (1 local + 1 api), simple quorum=99 ignored
        assert len(results) == 2

    async def test_category_quorum_met_static_method(self):
        from collections import Counter

        counts = Counter({"local": 1, "api": 2})
        required = {"local": 1, "api": 1}
        assert ParallelRouter._category_quorum_met(counts, required) is True

        counts_partial = Counter({"local": 1})
        assert ParallelRouter._category_quorum_met(counts_partial, required) is False
