"""Tests for batch adapters."""
from __future__ import annotations

from injection_guard.eval.batch import (
    AnthropicBatchAdapter,
    GeminiBatchAdapter,
    OpenAIBatchAdapter,
)


class TestAnthropicBatchAdapter:
    """Test the AnthropicBatchAdapter construction."""

    def test_construction(self):
        adapter = AnthropicBatchAdapter(api_key="test-key", model="claude-sonnet-4-20250514")
        assert adapter._api_key == "test-key"
        assert adapter.model == "claude-sonnet-4-20250514"

    def test_default_parameters(self):
        adapter = AnthropicBatchAdapter()
        assert adapter.model == "claude-sonnet-4-20250514"


class TestOpenAIBatchAdapter:
    """Test the OpenAIBatchAdapter construction."""

    def test_construction(self):
        adapter = OpenAIBatchAdapter(api_key="test-key", model="gpt-4o")
        assert adapter._api_key == "test-key"
        assert adapter.model == "gpt-4o"

    def test_default_parameters(self):
        adapter = OpenAIBatchAdapter()
        assert adapter.model == "gpt-4o"


class TestGeminiBatchAdapter:
    """Test the GeminiBatchAdapter construction."""

    def test_construction(self):
        adapter = GeminiBatchAdapter(api_key="test-key", model="gemini-2.0-flash")
        assert adapter._api_key == "test-key"
        assert adapter.model == "gemini-2.0-flash"

    def test_default_parameters(self):
        adapter = GeminiBatchAdapter()
        assert adapter.model == "gemini-2.0-flash"
