"""Preprocessor package for injection-guard.

Exports the main ``Preprocessor`` pipeline class alongside the key output
types re-exported from ``injection_guard.types`` for convenience.
"""
from __future__ import annotations

from injection_guard.types import PreprocessorOutput, SignalVector
from injection_guard.preprocessor.pipeline import Preprocessor

__all__ = ["Preprocessor", "PreprocessorOutput", "SignalVector"]
