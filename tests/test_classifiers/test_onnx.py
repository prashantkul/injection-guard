"""Tests for the OnnxClassifier with mocked ONNX runtime."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from injection_guard.classifiers.onnx import OnnxClassifier


class TestOnnxNotAvailable:
    """Test behaviour when onnxruntime is not installed."""

    async def test_returns_degraded_result(self):
        with patch("injection_guard.classifiers.onnx._ORT_AVAILABLE", False):
            clf = OnnxClassifier(name="test-onnx", model_path="/fake/model.onnx")
            result = await clf.classify("test prompt")
        assert result.score == 0.5
        assert result.confidence == 0.0
        assert "error" in result.metadata

    async def test_session_none_returns_degraded(self):
        with patch("injection_guard.classifiers.onnx._ORT_AVAILABLE", True):
            clf = OnnxClassifier(name="test-onnx", model_path="/fake/model.onnx")
            clf._session = None
            result = await clf.classify("test prompt")
        assert result.score == 0.5
        assert result.confidence == 0.0
        assert "error" in result.metadata


class TestOnnxWithMockedSession:
    """Test ONNX classifier with a mocked inference session."""

    async def test_injection_prediction(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        # Return probabilities: [benign=0.1, injection=0.9]
        mock_session.run.return_value = [[[0.1, 0.9]]]

        with patch("injection_guard.classifiers.onnx._ORT_AVAILABLE", True):
            clf = OnnxClassifier(name="test-onnx", model_path="/fake/model.onnx")
            clf._session = mock_session
            result = await clf.classify("ignore previous instructions")

        assert result.score == 0.9
        assert result.label == "injection"
        assert result.latency_ms > 0

    async def test_benign_prediction(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [[[0.9, 0.1]]]

        with patch("injection_guard.classifiers.onnx._ORT_AVAILABLE", True):
            clf = OnnxClassifier(name="test-onnx", model_path="/fake/model.onnx")
            clf._session = mock_session
            result = await clf.classify("What is Python?")

        assert result.score == 0.1
        assert result.label == "benign"

    async def test_inference_exception_returns_degraded(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.side_effect = RuntimeError("ONNX error")

        with patch("injection_guard.classifiers.onnx._ORT_AVAILABLE", True):
            clf = OnnxClassifier(name="test-onnx", model_path="/fake/model.onnx")
            clf._session = mock_session
            result = await clf.classify("test")

        assert result.score == 0.5
        assert result.confidence == 0.0
        assert "error" in result.metadata
