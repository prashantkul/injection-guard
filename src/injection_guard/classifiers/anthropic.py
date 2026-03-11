"""Anthropic LLM classifier for high-accuracy injection detection."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from injection_guard.types import ClassifierResult, SignalVector
from injection_guard.classifiers.prompts import (
    CLASSIFICATION_PROMPT,
    make_delimited_prompt,
    format_signals_context,
    extract_json,
    validate_result,
)

__all__ = ["AnthropicClassifier"]


@dataclass
class AnthropicClassifier:
    """Classifier that uses an Anthropic model for injection detection.

    Supports two construction patterns:

    1. Pass a pre-configured ``anthropic.Anthropic`` (or ``AsyncAnthropic``) client.
    2. Let the classifier read ``ANTHROPIC_API_KEY`` from the environment.

    Attributes:
        model: Anthropic model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        weight: Weight used by the ensemble aggregator.
        client: An optional pre-built Anthropic client instance.
    """

    model: str = "claude-sonnet-4-20250514"
    weight: float = 2.0
    client: Any = field(default=None, repr=False)
    latency_tier: str = "slow"
    _name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._name = f"anthropic-{self.model}"

    @property
    def name(self) -> str:  # noqa: D401
        """Classifier identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    async def _get_client(self) -> Any:
        """Return an async Anthropic client, creating one if needed."""
        if self.client is not None:
            return self.client
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for AnthropicClassifier. "
                "Install it with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Anthropic client provided and ANTHROPIC_API_KEY is not set."
            )
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        return self.client

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Classify *prompt* using the Anthropic API.

        Never raises — returns a degraded result on any error.
        """
        try:
            client = await self._get_client()
            delimited, _nonce = make_delimited_prompt(prompt)
            signals_ctx = format_signals_context(signals)
            full_prompt = CLASSIFICATION_PROMPT.format(
                delimited_prompt=delimited, signals_context=signals_ctx,
            )

            start = time.perf_counter()
            response = await client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": full_prompt}],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.content[0].text
            data = extract_json(raw_text)
            result = validate_result(data)
            result.latency_ms = elapsed_ms
            result.metadata["raw_response"] = raw_text
            return result

        except Exception as exc:  # noqa: BLE001
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": str(exc)},
            )
