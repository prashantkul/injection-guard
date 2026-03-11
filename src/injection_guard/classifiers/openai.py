"""OpenAI LLM classifier for injection detection."""
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

__all__ = ["OpenAIClassifier"]


@dataclass
class OpenAIClassifier:
    """Classifier that uses an OpenAI model for injection detection.

    Supports two construction patterns:

    1. Pass a pre-configured ``openai.AsyncOpenAI`` client.
    2. Let the classifier read ``OPENAI_API_KEY`` from the environment.

    Attributes:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``).
        weight: Weight used by the ensemble aggregator.
        client: An optional pre-built OpenAI client instance.
    """

    model: str = "gpt-4o"
    weight: float = 1.5
    client: Any = field(default=None, repr=False)
    latency_tier: str = "medium"
    reasoning_effort: str | None = field(default=None)
    _name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._name = f"openai-{self.model}"

    @property
    def name(self) -> str:  # noqa: D401
        """Classifier identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    async def _get_client(self) -> Any:
        """Return an async OpenAI client, creating one if needed."""
        if self.client is not None:
            return self.client
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for OpenAIClassifier. "
                "Install it with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No OpenAI client provided and OPENAI_API_KEY is not set."
            )
        self.client = openai.AsyncOpenAI(api_key=api_key)
        return self.client

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Classify *prompt* using the OpenAI API.

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
            # gpt-5 family uses reasoning tokens and doesn't support temperature=0
            is_gpt5 = "gpt-5" in self.model
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_completion_tokens": 256,
                "messages": [
                    {"role": "system", "content": "You are a prompt-injection classifier."},
                    {"role": "user", "content": full_prompt},
                ],
            }
            if not is_gpt5:
                kwargs["temperature"] = 0
            if self.reasoning_effort or is_gpt5:
                kwargs["reasoning_effort"] = self.reasoning_effort or "medium"
                kwargs["max_completion_tokens"] = 1024
            response = await client.chat.completions.create(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.choices[0].message.content or ""
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
