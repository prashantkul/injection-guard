"""Local LLM classifier for OpenAI-compatible servers (Ollama, vLLM, etc.)."""
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

__all__ = ["LocalLLMClassifier"]


@dataclass
class LocalLLMClassifier:
    """Classifier for OpenAI-compatible local LLM servers.

    Works with Ollama, vLLM, llama.cpp, LM Studio, or any server that
    exposes the ``/v1/chat/completions`` endpoint.

    Supports three construction patterns:

    1. Pass a pre-configured ``openai.AsyncOpenAI`` client.
    2. Provide ``base_url`` and the classifier creates the client.
    3. Read ``LOCAL_LLM_BASE_URL`` from the environment.

    Attributes:
        model: Model name as known by the server (e.g. ``"llama3.2"``).
        base_url: Server URL (e.g. ``"http://localhost:11434/v1"`` for Ollama).
        weight: Weight used by the ensemble aggregator.
        client: An optional pre-built OpenAI-compatible client instance.

    Examples::

        # Ollama
        clf = LocalLLMClassifier(
            model="llama3.2",
            base_url="http://localhost:11434/v1",
        )

        # vLLM
        clf = LocalLLMClassifier(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )

        # From environment (LOCAL_LLM_BASE_URL + LOCAL_LLM_MODEL)
        clf = LocalLLMClassifier()
    """

    model: str = ""
    base_url: str = ""
    weight: float = 1.5
    client: Any = field(default=None, repr=False)
    latency_tier: str = "medium"
    api_key: str = "not-needed"
    _name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
        if not self.base_url:
            self.base_url = os.environ.get(
                "LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"
            )
        self._name = f"local-{self.model}"

    @property
    def name(self) -> str:  # noqa: D401
        """Classifier identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    async def _get_client(self) -> Any:
        """Return an async OpenAI-compatible client, creating one if needed."""
        if self.client is not None:
            return self.client
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for LocalLLMClassifier. "
                "Install it with: pip install openai"
            ) from exc

        self.client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return self.client

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Classify *prompt* using a local LLM server.

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
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a prompt-injection classifier."},
                    {"role": "user", "content": full_prompt},
                ],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.choices[0].message.content or ""
            data = extract_json(raw_text)
            result = validate_result(data)
            result.latency_ms = elapsed_ms
            result.metadata["raw_response"] = raw_text
            result.metadata["base_url"] = self.base_url
            return result

        except Exception as exc:  # noqa: BLE001
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": str(exc), "base_url": self.base_url},
            )
