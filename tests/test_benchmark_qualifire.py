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

import os
import random
from collections import Counter

import pytest

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from injection_guard.guard import InjectionGuard
from injection_guard.classifiers.regex import RegexPrefilter
from injection_guard.router.cascade import CascadeRouter
from injection_guard.types import Action, CascadeConfig

DATASET_ID = "qualifire/prompt-injections-benchmark"
SAMPLE_SIZE = 100


def _get_hf_token() -> str | None:
    """Get HF token from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("HF_TOKEN")
    except ImportError:
        return None


def _load_qualifire(split: str = "test", sample_size: int | None = None):
    """Load the Qualifire dataset, optionally sampling."""
    token = _get_hf_token()
    ds = load_dataset(DATASET_ID, split=split, token=token)
    if sample_size and len(ds) > sample_size:
        indices = random.Random(42).sample(range(len(ds)), sample_size)
        ds = ds.select(indices)
    return ds


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
        _load_qualifire(split="test[:1]")
        return True
    except Exception:
        return False


_skip_no_access = pytest.mark.skipif(
    not _can_access_dataset(),
    reason=f"Cannot access {DATASET_ID} (check HF_TOKEN and dataset access)",
)

pytestmark = [pytest.mark.benchmark, _skip_no_datasets, _skip_no_access]


class TestQualifireDatasetLoading:
    """Verify dataset loads correctly and has expected schema."""

    def test_dataset_has_expected_columns(self):
        ds = _load_qualifire(split="test[:5]")
        assert "text" in ds.column_names
        assert "label" in ds.column_names

    def test_labels_are_valid(self):
        ds = _load_qualifire(split="test[:50]")
        labels = set(ds["label"])
        assert labels <= {"jailbreak", "benign"}, f"Unexpected labels: {labels}"

    def test_dataset_has_both_classes(self):
        ds = _load_qualifire(split="test[:200]")
        labels = set(ds["label"])
        assert "jailbreak" in labels, "No jailbreak samples found"
        assert "benign" in labels, "No benign samples found"


class TestQualifireClassification:
    """Run injection-guard against Qualifire samples."""

    async def test_injection_scores_higher_than_benign(self):
        """Average score for jailbreak prompts exceeds average for benign."""
        ds = _load_qualifire(split="test", sample_size=SAMPLE_SIZE)
        guard = _make_guard()

        injection_scores = []
        benign_scores = []

        for row in ds:
            decision = await guard.classify(row["text"])
            if row["label"] == "jailbreak":
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
        ds = _load_qualifire(split="test", sample_size=SAMPLE_SIZE)
        guard = _make_guard()

        blocked_benign = []
        for row in ds:
            if row["label"] != "benign":
                continue
            decision = await guard.classify(row["text"])
            if decision.action == Action.BLOCK:
                blocked_benign.append(row["text"][:80])

        # Allow up to 5% false positive rate for regex-only classifier
        benign_count = sum(1 for r in ds if r["label"] == "benign")
        if benign_count == 0:
            pytest.skip("No benign samples in subset")

        fp_rate = len(blocked_benign) / benign_count
        assert fp_rate <= 0.05, (
            f"False positive rate {fp_rate:.1%} exceeds 5%. "
            f"Blocked benign samples: {blocked_benign[:5]}"
        )

    async def test_known_injections_detected(self):
        """At least some jailbreak prompts should be FLAG or BLOCK."""
        ds = _load_qualifire(split="test", sample_size=SAMPLE_SIZE)
        guard = _make_guard()

        detected = 0
        total_injection = 0

        for row in ds:
            if row["label"] != "jailbreak":
                continue
            total_injection += 1
            decision = await guard.classify(row["text"])
            if decision.action in (Action.FLAG, Action.BLOCK):
                detected += 1

        if total_injection == 0:
            pytest.skip("No jailbreak samples in subset")

        detection_rate = detected / total_injection
        # Regex-only won't catch everything, but should get some
        assert detection_rate > 0.05, (
            f"Detection rate {detection_rate:.1%} is too low. "
            f"Detected {detected}/{total_injection} injections."
        )

    async def test_action_distribution_is_reasonable(self):
        """Check that we get a mix of actions, not all one category."""
        ds = _load_qualifire(split="test", sample_size=SAMPLE_SIZE)
        guard = _make_guard()

        actions = []
        for row in ds:
            decision = await guard.classify(row["text"])
            actions.append(decision.action.value)

        counts = Counter(actions)
        # Should have at least ALLOW and one of FLAG/BLOCK
        assert "allow" in counts, f"No ALLOW decisions. Distribution: {counts}"
        assert len(counts) >= 2, (
            f"Only one action type produced. Distribution: {counts}"
        )


class TestQualifireMetrics:
    """Compute and assert on classification metrics."""

    async def test_precision_recall_summary(self):
        """Log precision/recall for the regex-only classifier against Qualifire.

        This is more of a reporting test — it always passes but prints metrics.
        """
        ds = _load_qualifire(split="test", sample_size=200)
        guard = _make_guard()

        tp = fp = tn = fn = 0

        for row in ds:
            decision = await guard.classify(row["text"])
            predicted_injection = decision.action in (Action.FLAG, Action.BLOCK)
            actual_injection = row["label"] == "jailbreak"

            if predicted_injection and actual_injection:
                tp += 1
            elif predicted_injection and not actual_injection:
                fp += 1
            elif not predicted_injection and actual_injection:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n{'='*50}")
        print(f"Qualifire Benchmark — RegexPrefilter only")
        print(f"{'='*50}")
        print(f"Samples: {len(ds)} (TP={tp} FP={fp} TN={tn} FN={fn})")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1:        {f1:.3f}")
        print(f"{'='*50}")

        # Regex-only: we expect low recall but decent precision
        assert precision >= 0.0  # always passes — metrics are informational
