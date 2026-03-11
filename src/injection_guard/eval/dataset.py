"""Dataset loaders for evaluation and integration testing.

Normalizes HuggingFace datasets into a common ``TestSample`` format.
Supports Qualifire (PI/JB) and ToxicChat (safety/toxicity).

Requires: ``pip install datasets`` (or ``injection-guard[benchmark]``).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TestSample",
    "load_qualifire",
    "load_toxicchat",
    "load_mixed",
]


@dataclass
class TestSample:
    """A single evaluation sample in normalized format.

    Attributes:
        prompt: The input text to classify.
        label: Normalized label — ``"injection"``, ``"benign"``, ``"toxic"``, or ``"safe"``.
        source: Dataset origin (e.g. ``"qualifire"``, ``"toxicchat"``).
        metadata: Original fields from the source dataset.
    """

    prompt: str
    label: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        """Whether this sample represents a harmful input (injection or toxic)."""
        return self.label in ("injection", "toxic")


def _get_hf_token() -> str | None:
    """Get HuggingFace token from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv()
        return os.environ.get("HF_TOKEN")
    except ImportError:
        return None


def load_qualifire(
    n: int | None = None,
    seed: int = 42,
    split: str = "test",
    balanced: bool = True,
) -> list[TestSample]:
    """Load samples from the Qualifire prompt-injections-benchmark dataset.

    Args:
        n: Total number of samples to return. None = all.
        seed: Random seed for reproducible sampling.
        split: Dataset split to load.
        balanced: If True and n is set, return n/2 injection + n/2 benign.

    Returns:
        List of TestSample with label ``"injection"`` or ``"benign"``.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    token = _get_hf_token()
    ds = load_dataset("qualifire/prompt-injections-benchmark", split=split, token=token)

    if balanced and n is not None:
        rng = random.Random(seed)
        injections = [r for r in ds if r["label"] == "jailbreak"]
        benign = [r for r in ds if r["label"] == "benign"]
        half = n // 2
        injections = rng.sample(injections, min(half, len(injections)))
        benign = rng.sample(benign, min(half, len(benign)))
        rows = injections + benign
        rng.shuffle(rows)
    elif n is not None:
        indices = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
        rows = [ds[i] for i in indices]
    else:
        rows = list(ds)

    return [
        TestSample(
            prompt=r["text"],
            label="injection" if r["label"] == "jailbreak" else "benign",
            source="qualifire",
            metadata={"original_label": r["label"]},
        )
        for r in rows
    ]


def load_toxicchat(
    n: int | None = None,
    seed: int = 42,
    split: str = "test",
    config: str = "toxicchat0124",
    balanced: bool = True,
) -> list[TestSample]:
    """Load samples from the lmsys/toxic-chat dataset.

    Args:
        n: Total number of samples to return. None = all.
        seed: Random seed for reproducible sampling.
        split: Dataset split to load.
        config: Dataset config (``"toxicchat0124"`` or ``"toxicchat1123"``).
        balanced: If True and n is set, return n/2 toxic + n/2 safe.

    Returns:
        List of TestSample with label ``"toxic"`` or ``"safe"``.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("lmsys/toxic-chat", config, split=split)

    if balanced and n is not None:
        rng = random.Random(seed)
        toxic = [r for r in ds if r["toxicity"] == 1]
        safe = [r for r in ds if r["toxicity"] == 0]
        half = n // 2
        toxic = rng.sample(toxic, min(half, len(toxic)))
        safe = rng.sample(safe, min(half, len(safe)))
        rows = toxic + safe
        rng.shuffle(rows)
    elif n is not None:
        indices = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
        rows = [ds[i] for i in indices]
    else:
        rows = list(ds)

    return [
        TestSample(
            prompt=r["user_input"],
            label="toxic" if r["toxicity"] == 1 else "safe",
            source="toxicchat",
            metadata={
                "toxicity": r["toxicity"],
                "jailbreaking": r.get("jailbreaking", 0),
                "human_annotation": r.get("human_annotation", False),
            },
        )
        for r in rows
    ]


def load_mixed(
    n_per_source: int = 50,
    seed: int = 42,
    balanced: bool = True,
) -> list[TestSample]:
    """Load a mixed dataset combining Qualifire (PI/JB) and ToxicChat (safety).

    Args:
        n_per_source: Number of samples to pull from each dataset.
        seed: Random seed for reproducible sampling.
        balanced: If True, balance attack/benign within each source.

    Returns:
        Combined list of TestSample from both datasets, shuffled.
    """
    qualifire = load_qualifire(n=n_per_source, seed=seed, balanced=balanced)
    toxicchat = load_toxicchat(n=n_per_source, seed=seed, balanced=balanced)

    combined = qualifire + toxicchat
    random.Random(seed).shuffle(combined)
    return combined
