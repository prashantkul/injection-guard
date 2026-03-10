"""EvalRunner — load datasets and run evaluation through InjectionGuard."""
from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path

from injection_guard.types import Decision, EvalSample

__all__ = ["EvalRunner"]


class EvalRunner:
    """Run evaluation datasets through an InjectionGuard instance.

    Args:
        guard: A configured ``InjectionGuard`` instance used to classify
            each prompt in the evaluation dataset.
    """

    def __init__(self, guard: object) -> None:
        # Avoid a hard import of InjectionGuard at module level so the
        # eval sub-package can be imported independently for analysis.
        from injection_guard.guard import InjectionGuard

        if not isinstance(guard, InjectionGuard):
            raise TypeError(
                f"guard must be an InjectionGuard instance, got {type(guard).__name__}"
            )
        self._guard: InjectionGuard = guard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        dataset: str,
        *,
        use_batch_api: bool = False,
        batch_size: int = 100,
    ) -> "EvalReport":
        """Run evaluation on a dataset and produce an ``EvalReport``.

        Args:
            dataset: Path to a JSONL or CSV file.  Each record must have
                a ``prompt`` field and a ``label`` field (``"injection"``
                or ``"benign"``).
            use_batch_api: If ``True``, use batch adapters for
                throughput (not yet implemented).
            batch_size: Number of prompts per batch when
                ``use_batch_api`` is ``True``.

        Returns:
            An ``EvalReport`` computed from the predictions.
        """
        from injection_guard.eval.report import EvalReport

        samples = self._load_dataset(dataset)

        if use_batch_api:
            predictions = await self._run_batched(samples, batch_size)
        else:
            predictions = await self._run_sequential(samples)

        return EvalReport(predictions)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(path: str) -> list[EvalSample]:
        """Load a JSONL or CSV dataset from *path*.

        The file must contain ``prompt`` and ``label`` columns/fields.

        Returns:
            List of ``EvalSample`` instances.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file format is unsupported or required
                columns are missing.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        ext = filepath.suffix.lower()

        if ext == ".jsonl":
            return EvalRunner._load_jsonl(filepath)
        elif ext == ".csv":
            return EvalRunner._load_csv(filepath)
        else:
            raise ValueError(
                f"Unsupported dataset format '{ext}'. Use .jsonl or .csv."
            )

    @staticmethod
    def _load_jsonl(filepath: Path) -> list[EvalSample]:
        """Load samples from a JSONL file."""
        samples: list[EvalSample] = []
        with open(filepath, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {filepath}: {exc}"
                    ) from exc

                prompt = record.get("prompt")
                label = record.get("label")
                if prompt is None or label is None:
                    raise ValueError(
                        f"Missing 'prompt' or 'label' on line {line_no} of {filepath}"
                    )
                samples.append(EvalSample(prompt=str(prompt), label=str(label)))
        return samples

    @staticmethod
    def _load_csv(filepath: Path) -> list[EvalSample]:
        """Load samples from a CSV file."""
        samples: list[EvalSample] = []
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {"prompt", "label"}.issubset(
                set(reader.fieldnames)
            ):
                raise ValueError(
                    f"CSV file {filepath} must have 'prompt' and 'label' columns. "
                    f"Found: {reader.fieldnames}"
                )
            for row_no, row in enumerate(reader, start=2):
                prompt = row.get("prompt")
                label = row.get("label")
                if prompt is None or label is None:
                    raise ValueError(
                        f"Missing 'prompt' or 'label' on row {row_no} of {filepath}"
                    )
                samples.append(EvalSample(prompt=str(prompt), label=str(label)))
        return samples

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    async def _run_sequential(
        self, samples: list[EvalSample]
    ) -> list[tuple[Decision, str]]:
        """Classify each sample sequentially."""
        predictions: list[tuple[Decision, str]] = []
        for sample in samples:
            decision = await self._guard.classify(sample.prompt)
            predictions.append((decision, sample.label))
        return predictions

    async def _run_batched(
        self, samples: list[EvalSample], batch_size: int
    ) -> list[tuple[Decision, str]]:
        """Classify samples in concurrent batches.

        When the batch API adapters are fully implemented this method
        will delegate to them.  For now it uses
        ``InjectionGuard.classify_batch`` for concurrency.
        """
        predictions: list[tuple[Decision, str]] = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            prompts = [s.prompt for s in chunk]
            decisions = await self._guard.classify_batch(prompts)
            for decision, sample in zip(decisions, chunk):
                predictions.append((decision, sample.label))
        return predictions
