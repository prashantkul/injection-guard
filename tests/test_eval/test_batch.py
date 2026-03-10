"""Tests for batch adapter stubs."""
from __future__ import annotations

import pytest

from injection_guard.eval.batch import AnthropicBatchAdapter, OpenAIBatchAdapter


class TestAnthropicBatchAdapter:
    """Test the AnthropicBatchAdapter stub."""

    def test_construction(self):
        adapter = AnthropicBatchAdapter(api_key="test-key", model="claude-sonnet-4-20250514")
        assert adapter._api_key == "test-key"
        assert adapter._model == "claude-sonnet-4-20250514"

    async def test_submit_batch_raises_not_implemented(self):
        adapter = AnthropicBatchAdapter()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await adapter.submit_batch(["prompt1", "prompt2"])

    async def test_collect_results_raises_not_implemented(self):
        adapter = AnthropicBatchAdapter()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await adapter.collect_results("batch-123")

    def test_default_parameters(self):
        adapter = AnthropicBatchAdapter()
        assert adapter._model == "claude-sonnet-4-20250514"
        assert adapter._max_concurrent == 5


class TestOpenAIBatchAdapter:
    """Test the OpenAIBatchAdapter stub."""

    def test_construction(self):
        adapter = OpenAIBatchAdapter(api_key="test-key", model="gpt-4o")
        assert adapter._api_key == "test-key"
        assert adapter._model == "gpt-4o"

    async def test_submit_batch_raises_not_implemented(self):
        adapter = OpenAIBatchAdapter()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await adapter.submit_batch(["prompt1", "prompt2"])

    async def test_collect_results_raises_not_implemented(self):
        adapter = OpenAIBatchAdapter()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await adapter.collect_results("batch-456")

    def test_default_parameters(self):
        adapter = OpenAIBatchAdapter()
        assert adapter._model == "gpt-4o"
        assert adapter._max_concurrent == 5
