"""Classifier modules for injection-guard."""
from __future__ import annotations

from injection_guard.types import BaseClassifier, ClassifierResult

from injection_guard.classifiers.regex import RegexPrefilter
from injection_guard.classifiers.onnx import OnnxClassifier
from injection_guard.classifiers.anthropic import AnthropicClassifier
from injection_guard.classifiers.openai import OpenAIClassifier

__all__ = [
    "BaseClassifier",
    "ClassifierResult",
    "RegexPrefilter",
    "OnnxClassifier",
    "AnthropicClassifier",
    "OpenAIClassifier",
]
