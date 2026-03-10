"""Batch API adapters for large-scale evaluation.

These are interface stubs — the real batch implementation is future work.
They define the contract that batch runners will follow.
"""
from __future__ import annotations

from injection_guard.types import ClassifierResult, SignalVector

__all__ = ["AnthropicBatchAdapter", "OpenAIBatchAdapter"]


class AnthropicBatchAdapter:
    """Adapter for the Anthropic Messages Batch API.

    This is a stub that defines the interface for future implementation.
    The real adapter will submit prompts to the Anthropic batch endpoint,
    poll for completion, and parse results.

    Args:
        api_key: Anthropic API key. If ``None``, reads from
            ``ANTHROPIC_API_KEY`` environment variable.
        model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        max_concurrent: Maximum number of concurrent batch requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 5,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_concurrent = max_concurrent

    async def submit_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        signals: list[SignalVector | None] | None = None,
    ) -> str:
        """Submit a batch of prompts for classification.

        Args:
            prompts: List of prompts to classify.
            system_prompt: Optional system prompt for the classifier.
            signals: Optional preprocessor signals for each prompt.

        Returns:
            A batch ID string that can be used to collect results.

        Raises:
            NotImplementedError: Always, until the real implementation
                is provided.
        """
        raise NotImplementedError(
            "AnthropicBatchAdapter.submit_batch is not yet implemented. "
            "This is a stub for the future Anthropic Batch API integration."
        )

    async def collect_results(
        self,
        batch_id: str,
        poll_interval_s: float = 30.0,
        timeout_s: float = 3600.0,
    ) -> list[ClassifierResult]:
        """Poll for and collect batch results.

        Args:
            batch_id: The batch ID returned by ``submit_batch``.
            poll_interval_s: Seconds between polling attempts.
            timeout_s: Maximum time to wait before raising a timeout.

        Returns:
            List of ``ClassifierResult`` objects, one per input prompt.

        Raises:
            NotImplementedError: Always, until the real implementation
                is provided.
        """
        raise NotImplementedError(
            "AnthropicBatchAdapter.collect_results is not yet implemented. "
            "This is a stub for the future Anthropic Batch API integration."
        )


class OpenAIBatchAdapter:
    """Adapter for the OpenAI Batch API.

    This is a stub that defines the interface for future implementation.
    The real adapter will submit JSONL files to the OpenAI batch endpoint,
    poll for completion, and parse results.

    Args:
        api_key: OpenAI API key. If ``None``, reads from
            ``OPENAI_API_KEY`` environment variable.
        model: Model identifier (e.g. ``"gpt-4o"``).
        max_concurrent: Maximum number of concurrent batch requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_concurrent: int = 5,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_concurrent = max_concurrent

    async def submit_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        signals: list[SignalVector | None] | None = None,
    ) -> str:
        """Submit a batch of prompts for classification.

        Args:
            prompts: List of prompts to classify.
            system_prompt: Optional system prompt for the classifier.
            signals: Optional preprocessor signals for each prompt.

        Returns:
            A batch ID string that can be used to collect results.

        Raises:
            NotImplementedError: Always, until the real implementation
                is provided.
        """
        raise NotImplementedError(
            "OpenAIBatchAdapter.submit_batch is not yet implemented. "
            "This is a stub for the future OpenAI Batch API integration."
        )

    async def collect_results(
        self,
        batch_id: str,
        poll_interval_s: float = 30.0,
        timeout_s: float = 3600.0,
    ) -> list[ClassifierResult]:
        """Poll for and collect batch results.

        Args:
            batch_id: The batch ID returned by ``submit_batch``.
            poll_interval_s: Seconds between polling attempts.
            timeout_s: Maximum time to wait before raising a timeout.

        Returns:
            List of ``ClassifierResult`` objects, one per input prompt.

        Raises:
            NotImplementedError: Always, until the real implementation
                is provided.
        """
        raise NotImplementedError(
            "OpenAIBatchAdapter.collect_results is not yet implemented. "
            "This is a stub for the future OpenAI Batch API integration."
        )
