"""Gate module for pre-screening prompts before classification."""
from __future__ import annotations

from injection_guard.gate.model_armor import ModelArmorGate
from injection_guard.types import ModelArmorResult

__all__ = ["ModelArmorGate", "ModelArmorResult"]
