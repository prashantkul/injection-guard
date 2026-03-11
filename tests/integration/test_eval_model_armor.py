"""Evaluation: Model Armor templates on Qualifire dataset.

Run:
  pytest tests/integration/test_eval_model_armor.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import random

import pytest

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from dotenv import load_dotenv
load_dotenv()

DATASET_ID = "qualifire/prompt-injections-benchmark"


def _load_qualifire(sample_size: int = 200) -> list[dict]:
    """Load Qualifire dataset, balanced sample."""
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(DATASET_ID, split="test", token=token)
    rng = random.Random(42)
    injections = [r for r in ds if r["label"] == "jailbreak"]
    benign = [r for r in ds if r["label"] == "benign"]
    n_each = sample_size // 2
    sampled = rng.sample(injections, min(n_each, len(injections)))
    sampled += rng.sample(benign, min(n_each, len(benign)))
    rng.shuffle(sampled)
    return sampled


def _can_access_dataset() -> bool:
    if not HAS_DATASETS:
        return False
    try:
        load_dataset(DATASET_ID, split="test[:1]", token=os.environ.get("HF_TOKEN"))
        return True
    except Exception:
        return False


_skip = pytest.mark.skipif(
    not _can_access_dataset(),
    reason=f"Cannot access {DATASET_ID}",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _metrics(tp: int, fn: int, tn: int, fp: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) > 0 else 0.0
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def _print_table(results: dict[str, dict]) -> None:
    _log(f"\n{'=' * 95}")
    _log(f"  {'Model':<35s} {'TP':>4s} {'FN':>4s} {'TN':>4s} {'FP':>4s}  {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Acc':>6s} {'Latency':>8s}")
    _log(f"  {'-'*35} {'-'*4} {'-'*4} {'-'*4} {'-'*4}  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for name, m in results.items():
        lat = f"{m.get('avg_latency_ms', 0):.0f}ms"
        _log(
            f"  {name:<35s} {m['tp']:>4d} {m['fn']:>4d} {m['tn']:>4d} {m['fp']:>4d}"
            f"  {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f} {lat:>8s}"
        )
    _log(f"{'=' * 95}")


async def _eval_model_armor(gate, samples: list[dict], name: str, concurrency: int = 10) -> dict:
    """Run Model Armor gate against samples in parallel."""
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(samples)

    async def _screen_one(row: dict) -> tuple[bool, bool, float]:
        nonlocal done
        expected = row["label"] in ("jailbreak", "injection")
        async with sem:
            result = await gate.screen(row["text"])
        done += 1
        if done % 50 == 0 or done == total:
            _log(f"    {name}: {done}/{total}")
        return expected, result.match_found, result.latency_ms

    tasks = [_screen_one(row) for row in samples]
    outcomes = await asyncio.gather(*tasks)

    tp = fn = tn = fp = 0
    total_ms = 0.0
    for expected, predicted, lat in outcomes:
        total_ms += lat
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fp += 1

    m = _metrics(tp, fn, tn, fp)
    m["avg_latency_ms"] = total_ms / len(samples) if samples else 0
    return m


@_skip
class TestEvalModelArmor:
    """Evaluate Model Armor templates on Qualifire dataset."""

    @pytest.mark.timeout(600)
    async def test_model_armor_200_samples(self):
        from injection_guard.gate.model_armor import ModelArmorGate

        samples = _load_qualifire(sample_size=200)
        n_inj = sum(1 for s in samples if s["label"] in ("jailbreak", "injection"))
        _log(f"\nDataset: {len(samples)} samples ({n_inj} injection, {len(samples) - n_inj} benign)")

        results: dict[str, dict] = {}
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = os.environ.get("MA_LOCATION", "us-central1")

        for tmpl_env, tmpl_name, block_on in [
            ("MA_HIGH_TEMPLATE_ID", "MA High", "HIGH"),
            ("MA_MEDIUM_TEMPLATE_ID", "MA Medium", "MEDIUM_AND_ABOVE"),
            ("MA_LOW_TEMPLATE_ID", "MA Low", "LOW_AND_ABOVE"),
        ]:
            tmpl_id = os.environ.get(tmpl_env, "")
            if not tmpl_id or not project:
                _log(f"  Skipping {tmpl_name} — env not set")
                continue
            gate = ModelArmorGate(
                project_id=project,
                location=location,
                template_id=tmpl_id,
                block_on=block_on,
                fail_mode="open",
            )
            _log(f"  Running {tmpl_name}...")
            results[tmpl_name] = await _eval_model_armor(gate, samples, tmpl_name)
            _log(f"  Done {tmpl_name}: acc={results[tmpl_name]['accuracy']:.3f}")

        _print_table(results)
        assert results, "No Model Armor templates were tested"
