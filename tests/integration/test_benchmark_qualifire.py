"""Benchmark tests using the Qualifire prompt-injections-benchmark dataset.

Dataset: https://huggingface.co/datasets/qualifire/prompt-injections-benchmark
Schema:  text (str), label ("jailbreak" | "benign")
Size:    ~5,000 samples

These tests download the dataset from HuggingFace and run a sample through
the injection-guard pipeline with the RegexPrefilter classifier. They verify
that the library produces meaningfully different scores for injection vs
benign prompts.

Requirements:
  - pip install datasets (or injection-guard[benchmark])
  - HF_TOKEN with gated-repo access in .env or environment
  - Accept dataset terms at the HuggingFace dataset page

Run:
  pytest tests/test_benchmark_qualifire.py -v
  pytest tests/test_benchmark_qualifire.py -v -k "not slow"  # skip full dataset
"""
from __future__ import annotations

from collections import Counter

import pytest

try:
    from injection_guard.eval.dataset import load_qualifire, TestSample
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from injection_guard.guard import InjectionGuard
from injection_guard.classifiers.regex import RegexPrefilter
from injection_guard.router.cascade import CascadeRouter
from injection_guard.types import Action, CascadeConfig

SAMPLE_SIZE = 100


def _make_guard() -> InjectionGuard:
    """Build a guard with RegexPrefilter for benchmarking."""
    cfg = CascadeConfig(timeout_ms=5000, max_retries=0)
    return InjectionGuard(
        classifiers=[RegexPrefilter()],
        router=CascadeRouter(cfg),
        thresholds={"block": 0.85, "flag": 0.50},
    )


_skip_no_datasets = pytest.mark.skipif(
    not HAS_DATASETS,
    reason="'datasets' package not installed (pip install datasets)",
)


def _can_access_dataset() -> bool:
    """Check if we can actually load the dataset."""
    if not HAS_DATASETS:
        return False
    try:
        load_qualifire(n=1, seed=42)
        return True
    except Exception:
        return False


_skip_no_access = pytest.mark.skipif(
    not _can_access_dataset(),
    reason="Cannot access qualifire/prompt-injections-benchmark (check HF_TOKEN and dataset access)",
)

pytestmark = [pytest.mark.benchmark, _skip_no_datasets, _skip_no_access]


class TestQualifireDatasetLoading:
    """Verify dataset loads correctly and has expected schema."""

    def test_dataset_has_both_classes(self):
        samples = load_qualifire(n=50, seed=42)
        labels = {s.label for s in samples}
        assert "injection" in labels, "No injection samples found"
        assert "benign" in labels, "No benign samples found"

    def test_samples_have_prompts(self):
        samples = load_qualifire(n=5, seed=42)
        for s in samples:
            assert len(s.prompt) > 0, "Empty prompt"
            assert s.source == "qualifire"


class TestQualifireClassification:
    """Run injection-guard against Qualifire samples."""

    async def test_injection_scores_higher_than_benign(self):
        """Average score for injection prompts exceeds average for benign."""
        samples = load_qualifire(n=SAMPLE_SIZE, seed=42)
        guard = _make_guard()

        injection_scores = []
        benign_scores = []

        for s in samples:
            decision = await guard.classify(s.prompt)
            if s.label == "injection":
                injection_scores.append(decision.ensemble_score)
            else:
                benign_scores.append(decision.ensemble_score)

        if not injection_scores or not benign_scores:
            pytest.skip("Sample didn't contain both classes")

        avg_injection = sum(injection_scores) / len(injection_scores)
        avg_benign = sum(benign_scores) / len(benign_scores)

        assert avg_injection > avg_benign, (
            f"Injection avg ({avg_injection:.3f}) should exceed "
            f"benign avg ({avg_benign:.3f})"
        )

    async def test_no_benign_blocked(self):
        """Benign prompts should never be BLOCK (low false-positive rate)."""
        samples = load_qualifire(n=SAMPLE_SIZE, seed=42)
        guard = _make_guard()

        benign = [s for s in samples if s.label == "benign"]
        if not benign:
            pytest.skip("No benign samples in subset")

        blocked_benign = []
        for s in benign:
            decision = await guard.classify(s.prompt)
            if decision.action == Action.BLOCK:
                blocked_benign.append(s.prompt[:80])

        fp_rate = len(blocked_benign) / len(benign)
        assert fp_rate <= 0.05, (
            f"False positive rate {fp_rate:.1%} exceeds 5%. "
            f"Blocked benign samples: {blocked_benign[:5]}"
        )

    async def test_known_injections_detected(self):
        """At least some injection prompts should be FLAG or BLOCK."""
        samples = load_qualifire(n=SAMPLE_SIZE, seed=42)
        guard = _make_guard()

        injections = [s for s in samples if s.label == "injection"]
        if not injections:
            pytest.skip("No injection samples in subset")

        detected = 0
        for s in injections:
            decision = await guard.classify(s.prompt)
            if decision.action in (Action.FLAG, Action.BLOCK):
                detected += 1

        detection_rate = detected / len(injections)
        assert detection_rate > 0.05, (
            f"Detection rate {detection_rate:.1%} is too low. "
            f"Detected {detected}/{len(injections)} injections."
        )

    async def test_action_distribution_is_reasonable(self):
        """Check that we get a mix of actions, not all one category."""
        samples = load_qualifire(n=SAMPLE_SIZE, seed=42)
        guard = _make_guard()

        actions = []
        for s in samples:
            decision = await guard.classify(s.prompt)
            actions.append(decision.action.value)

        counts = Counter(actions)
        assert "allow" in counts, f"No ALLOW decisions. Distribution: {counts}"
        assert len(counts) >= 2, (
            f"Only one action type produced. Distribution: {counts}"
        )


class TestQualifireMetrics:
    """Compute and assert on classification metrics."""

    async def test_precision_recall_summary(self):
        """Full benchmark report with Rich confusion matrix and metrics."""
        from injection_guard.reporting import print_benchmark

        samples = load_qualifire(n=200, seed=42)
        guard = _make_guard()

        decisions = []
        labels = []
        for s in samples:
            decision = await guard.classify(s.prompt)
            decisions.append(decision)
            # print_benchmark expects original qualifire labels
            labels.append(s.metadata.get("original_label", s.label))

        print_benchmark(
            decisions,
            labels,
            title="Qualifire Benchmark — RegexPrefilter only",
        )

        assert len(decisions) == len(labels)
