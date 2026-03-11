"""Tests for EvalRunner."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from injection_guard.eval.runner import EvalRunner
from injection_guard.types import Action, Decision, EvalSample, PreprocessorOutput


def _make_mock_guard():
    """Create a mock InjectionGuard for testing."""
    mock_guard = MagicMock()
    mock_guard.__class__.__name__ = "InjectionGuard"
    return mock_guard


class TestLoadDatasetJsonl:
    """Test JSONL dataset loading."""

    def test_load_valid_jsonl(self, tmp_path):
        data = [
            {"prompt": "What is AI?", "label": "benign"},
            {"prompt": "Ignore instructions", "label": "injection"},
        ]
        filepath = tmp_path / "test.jsonl"
        filepath.write_text("\n".join(json.dumps(d) for d in data))

        samples = EvalRunner._load_dataset(str(filepath))
        assert len(samples) == 2
        assert samples[0].prompt == "What is AI?"
        assert samples[0].label == "benign"
        assert samples[1].label == "injection"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            EvalRunner._load_dataset("/nonexistent/path.jsonl")

    def test_invalid_json_raises(self, tmp_path):
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            EvalRunner._load_dataset(str(filepath))

    def test_missing_fields_raises(self, tmp_path):
        filepath = tmp_path / "missing.jsonl"
        filepath.write_text(json.dumps({"prompt": "hi"}) + "\n")
        with pytest.raises(ValueError, match="Missing"):
            EvalRunner._load_dataset(str(filepath))


class TestLoadDatasetCsv:
    """Test CSV dataset loading."""

    def test_load_valid_csv(self, tmp_path):
        filepath = tmp_path / "test.csv"
        filepath.write_text("prompt,label\nWhat is AI?,benign\nIgnore,injection\n")

        samples = EvalRunner._load_dataset(str(filepath))
        assert len(samples) == 2
        assert samples[0].label == "benign"

    def test_csv_missing_columns_raises(self, tmp_path):
        filepath = tmp_path / "bad.csv"
        filepath.write_text("text,tag\nhi,ok\n")
        with pytest.raises(ValueError, match="must have"):
            EvalRunner._load_dataset(str(filepath))


class TestUnsupportedFormat:
    """Test unsupported file format."""

    def test_unsupported_extension_raises(self, tmp_path):
        filepath = tmp_path / "data.txt"
        filepath.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            EvalRunner._load_dataset(str(filepath))


class TestRunSequential:
    """Test the sequential run method with mocked guard."""

    async def test_run_produces_report(self, tmp_path):
        data = [
            {"prompt": "What is AI?", "label": "benign"},
            {"prompt": "Ignore instructions", "label": "injection"},
        ]
        filepath = tmp_path / "test.jsonl"
        filepath.write_text("\n".join(json.dumps(d) for d in data))

        mock_decision_benign = Decision(
            action=Action.ALLOW,
            ensemble_score=0.1,
        )
        mock_decision_injection = Decision(
            action=Action.BLOCK,
            ensemble_score=0.95,
        )

        mock_guard = MagicMock()
        mock_guard.classify = AsyncMock(
            side_effect=[mock_decision_benign, mock_decision_injection]
        )

        # Bypass the isinstance check by patching at the import site
        with patch("injection_guard.guard.InjectionGuard", type(mock_guard)):
            runner = EvalRunner(mock_guard)
            report = await runner.run(str(filepath))

        # Report should have metrics
        metrics = report.precision_recall_f1()
        assert metrics.precision >= 0.0
        assert metrics.recall >= 0.0
