"""Google Gemini classifier via Vertex AI for injection detection."""
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

__all__ = ["GeminiClassifier"]


@dataclass
class GeminiClassifier:
    """Classifier that uses Google Gemini via Vertex AI for injection detection.

    Supports two construction patterns:

    1. Pass a pre-configured ``GenerativeModel`` instance as ``client``.
    2. Let the classifier read ``GCP_PROJECT_ID`` and ``GCP_REGION`` from
       the environment and initialise Vertex AI automatically.

    Attributes:
        model: Gemini model identifier (e.g. ``"gemini-2.0-flash"``).
        weight: Weight used by the ensemble aggregator.
        client: An optional pre-built ``GenerativeModel`` instance.
        project: GCP project ID. Falls back to ``GCP_PROJECT_ID`` env var.
        region: GCP region. Falls back to ``GCP_REGION`` env var.
    """

    model: str = "gemini-2.0-flash"
    weight: float = 1.5
    client: Any = field(default=None, repr=False)
    latency_tier: str = "medium"
    project: str | None = None
    region: str | None = None
    _name: str = field(default="", init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._name = f"gemini-{self.model}"

    @property
    def name(self) -> str:  # noqa: D401
        """Classifier identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    async def _get_client(self) -> Any:
        """Return a Vertex AI GenerativeModel, creating one if needed."""
        if self.client is not None:
            return self.client
        try:
            import vertexai  # type: ignore[import-untyped]
            from vertexai.generative_models import GenerativeModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-cloud-aiplatform' package is required for GeminiClassifier. "
                "Install it with: pip install google-cloud-aiplatform"
            ) from exc

        project = self.project or os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = self.region or os.environ.get("GCP_REGION") or os.environ.get("GOOGLE_CLOUD_REGION", "global")

        if not project:
            raise RuntimeError(
                "No GCP project provided. Set 'project' parameter or GCP_PROJECT_ID env var."
            )

        if not self._initialized:
            vertexai.init(project=project, location=region)
            self._initialized = True

        self.client = GenerativeModel(self.model)
        return self.client

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Classify *prompt* using Google Gemini via Vertex AI.

        Never raises — returns a degraded result on any error.
        """
        try:
            model = await self._get_client()
            delimited, _nonce = make_delimited_prompt(prompt)
            signals_ctx = format_signals_context(signals)
            full_prompt = CLASSIFICATION_PROMPT.format(
                delimited_prompt=delimited, signals_context=signals_ctx,
            )

            start = time.perf_counter()
            response = await model.generate_content_async(
                full_prompt,
                generation_config={"temperature": 0, "max_output_tokens": 256},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.text
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
