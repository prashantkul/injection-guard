"""HuggingFace model classifier via OpenAI-compatible endpoint.

Connects to any server (litguard, vLLM, TGI) that serves HuggingFace
classification models behind an OpenAI-compatible chat completions API.

The server is expected to return a JSON object in message.content:
  {"label": "injection"|"benign", "score": 0.0-1.0, "confidence": 0.0-1.0}

Requires: ``pip install openai``
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from injection_guard.types import ClassifierResult, SignalVector

__all__ = ["HFCompatClassifier"]

_DEFAULT_BASE_URL = "http://localhost:8234/v1"


@dataclass
class HFCompatClassifier:
    """Classifier using a HuggingFace model served via OpenAI-compatible API.

    Designed for litguard or any server that wraps HF text-classification
    models (e.g. deepset/deberta-v3-base-injection, protectai/deberta-v3-base-prompt-injection-v2)
    behind ``/v1/chat/completions``.

    Attributes:
        model: Model name as known by the server.
        base_url: Server URL for OpenAI-compatible API.
        weight: Weight used by the ensemble aggregator.
    """

    model: str = "deberta-injection"
    base_url: str = ""
    weight: float = 1.0
    latency_tier: str = "fast"
    api_key: str = "not-needed"
    client: Any = field(default=None, repr=False)
    _name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("LITGUARD_BASE_URL", _DEFAULT_BASE_URL)
        self._name = f"hf-{self.model}"

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
                "The 'openai' package is required for HFCompatClassifier. "
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
        """Classify *prompt* using the HF model endpoint.

        Never raises — returns a degraded result on any error.
        """
        try:
            client = await self._get_client()

            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.choices[0].message.content or ""
            data = json.loads(raw_text)

            label = data.get("label", "injection")
            score = float(data.get("score", 0.5))
            confidence = float(data.get("confidence", score))

            # For HF models, score is P(LEGIT) or P(INJECTION) depending on model.
            # litguard normalizes: score is always confidence in the returned label.
            # Map to injection score (0=benign, 1=injection).
            if label == "benign":
                injection_score = 1.0 - score
            else:
                injection_score = score

            return ClassifierResult(
                score=injection_score,
                label=label,
                confidence=confidence,
                latency_ms=elapsed_ms,
                metadata={
                    "raw_response": raw_text,
                    "base_url": self.base_url,
                },
            )

        except Exception as exc:  # noqa: BLE001
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": str(exc), "base_url": self.base_url},
            )
