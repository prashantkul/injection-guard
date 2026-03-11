"""Tests for the CascadeRouter."""
from __future__ import annotations

import asyncio

import pytest

from injection_guard.router.cascade import CascadeRouter
from injection_guard.types import CascadeConfig, ClassifierResult


@pytest.fixture
def config() -> CascadeConfig:
    return CascadeConfig(
        timeout_ms=1000,
        max_retries=1,
        backoff_base_ms=10,
        fast_confidence=0.85,
        escalate_on_high_risk_prior=True,
        risk_prior_escalation_threshold=0.7,
    )


@pytest.fixture
def router(config: CascadeConfig) -> CascadeRouter:
    return CascadeRouter(config)


class TestFastClassifierConfident:
    """Test short-circuit when fast classifier is confident."""

    async def test_confident_benign_stops_early(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.05, label="benign", tier="fast")
        medium = make_classifier(name="medium", score=0.8, label="injection", tier="medium")
        slow = make_classifier(name="slow", score=0.9, label="injection", tier="slow")

        route = await router.route([fast, medium, slow], "benign text")

        # Fast returned score=0.05 < (1 - 0.85) = 0.15 -> confident benign
        assert len(route.results) == 1
        assert route.results[0][0] == "fast"
        assert route.quorum_met is True

    async def test_confident_injection_stops_early(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.95, label="injection", tier="fast")
        medium = make_classifier(name="medium", score=0.1, label="benign", tier="medium")

        route = await router.route([fast, medium], "attack text")

        # Fast returned score=0.95 > 0.85 -> confident injection
        assert len(route.results) == 1
        assert route.results[0][0] == "fast"
        assert route.quorum_met is True


class TestUncertainEscalates:
    """Test escalation when fast classifier is uncertain."""

    async def test_uncertain_escalates_to_medium(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.50, label="injection", tier="fast")
        medium = make_classifier(
            name="medium", score=0.95, label="injection", tier="medium"
        )

        route = await router.route([fast, medium], "ambiguous text")

        # Fast score=0.50 is not confident (0.15 < 0.50 < 0.85)
        # Medium score=0.95 > 0.85 -> confident, stops
        assert len(route.results) == 2
        assert route.results[0][0] == "fast"
        assert route.results[1][0] == "medium"
        assert route.quorum_met is True


class TestClassifierFailure:
    """Test that classifier failure doesn't stop the cascade."""

    async def test_failure_continues_to_next(self, router, make_classifier):
        failing = make_classifier(name="failing-fast", score=0.0, tier="fast", should_fail=True)
        good = make_classifier(name="good-medium", score=0.95, label="injection", tier="medium")

        route = await router.route([failing, good], "test text")

        # Failing classifier is skipped, good classifier runs
        assert len(route.results) == 1
        assert route.results[0][0] == "good-medium"


class TestHighRiskPriorSkipsFast:
    """Test that high risk_prior skips the fast tier."""

    async def test_skips_fast_tier(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.05, label="benign", tier="fast")
        medium = make_classifier(
            name="medium", score=0.95, label="injection", tier="medium"
        )

        route = await router.route(
            [fast, medium], "suspicious text", risk_prior=0.8
        )

        # Fast should be skipped due to risk_prior > 0.7
        names = [name for name, _ in route.results]
        assert "fast" not in names
        assert "medium" in names

    async def test_low_risk_prior_includes_fast(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.05, label="benign", tier="fast")
        medium = make_classifier(name="medium", score=0.5, label="injection", tier="medium")

        route = await router.route(
            [fast, medium], "normal text", risk_prior=0.3
        )

        # Fast should be included, and its confident result stops the cascade
        assert route.results[0][0] == "fast"


class TestTierOrdering:
    """Test that classifiers are executed in tier order."""

    async def test_fast_before_medium_before_slow(self, router, make_classifier):
        fast = make_classifier(name="fast", score=0.50, label="injection", tier="fast")
        slow = make_classifier(name="slow", score=0.50, label="injection", tier="slow")
        medium = make_classifier(name="medium", score=0.50, label="injection", tier="medium")

        # All are uncertain so all run
        route = await router.route([slow, fast, medium], "text")

        names = [name for name, _ in route.results]
        assert names == ["fast", "medium", "slow"]
