"""injection-guard: Ensemble prompt injection detection with pluggable classifiers."""
from __future__ import annotations

from injection_guard.types import (
    Action,
    Decision,
    ClassifierResult,
    PreprocessorOutput,
    SignalVector,
    ThresholdConfig,
    ModelArmorResult,
)
from injection_guard.guard import InjectionGuard
from injection_guard.router import CascadeRouter, ParallelRouter

from injection_guard.config import load_config

__all__ = [
    "InjectionGuard",
    "CascadeRouter",
    "ParallelRouter",
    "Action",
    "Decision",
    "ClassifierResult",
    "PreprocessorOutput",
    "SignalVector",
    "ThresholdConfig",
    "ModelArmorResult",
    "load_config",
]

__version__ = "0.1.0"
