"""Classifier modules for injection-guard."""
from __future__ import annotations

from injection_guard.types import BaseClassifier, ClassifierResult

from injection_guard.classifiers.regex import RegexPrefilter
from injection_guard.classifiers.onnx import OnnxClassifier
from injection_guard.classifiers.anthropic import AnthropicClassifier
from injection_guard.classifiers.openai import OpenAIClassifier
from injection_guard.classifiers.gemini import GeminiClassifier
from injection_guard.classifiers.local_llm import LocalLLMClassifier
from injection_guard.classifiers.safeguard import SafeguardClassifier
from injection_guard.classifiers.hf_compat import HFCompatClassifier

__all__ = [
    "BaseClassifier",
    "ClassifierResult",
    "RegexPrefilter",
    "OnnxClassifier",
    "AnthropicClassifier",
    "OpenAIClassifier",
    "GeminiClassifier",
    "LocalLLMClassifier",
    "SafeguardClassifier",
    "HFCompatClassifier",
]
