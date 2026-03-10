"""YAML/dict-based configuration for InjectionGuard."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from injection_guard.types import (
    BaseClassifier,
    CascadeConfig,
    ParallelConfig,
)

__all__ = ["load_config", "build_from_config"]

_CLASSIFIER_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> None:
    """Lazily populate the classifier registry on first use."""
    if _CLASSIFIER_REGISTRY:
        return

    from injection_guard.classifiers.regex import RegexPrefilter
    from injection_guard.classifiers.onnx import OnnxClassifier
    from injection_guard.classifiers.anthropic import AnthropicClassifier
    from injection_guard.classifiers.openai import OpenAIClassifier
    from injection_guard.classifiers.gemini import GeminiClassifier

    _CLASSIFIER_REGISTRY.update({
        "regex": RegexPrefilter,
        "onnx": OnnxClassifier,
        "anthropic": AnthropicClassifier,
        "openai": OpenAIClassifier,
        "gemini": GeminiClassifier,
    })


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _resolve_env(value: Any) -> Any:
    """Resolve ${ENV_VAR} or ${ENV_VAR:-default} references in string values."""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        inner = value[2:-1]
        if ":-" in inner:
            var_name, default = inner.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(inner, "")
    return value


def _build_classifier(cfg: dict[str, Any]) -> BaseClassifier:
    """Build a single classifier from its config dict.

    Args:
        cfg: Dict with at least a "type" key. All other keys are passed
             to the classifier constructor.

    Returns:
        An instantiated classifier.
    """
    _ensure_registry()

    clf_type = cfg.pop("type")
    if clf_type not in _CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier type '{clf_type}'. "
            f"Available: {list(_CLASSIFIER_REGISTRY.keys())}"
        )

    resolved = {k: _resolve_env(v) for k, v in cfg.items()}

    cls = _CLASSIFIER_REGISTRY[clf_type]
    return cls(**resolved)  # type: ignore[no-any-return]


def _build_router(cfg: dict[str, Any]) -> Any:
    """Build a router from its config dict."""
    from injection_guard.router import CascadeRouter, ParallelRouter

    router_type = cfg.pop("type", "cascade")
    resolved = {k: _resolve_env(v) for k, v in cfg.items()}

    if router_type == "cascade":
        config = CascadeConfig(**resolved)
        return CascadeRouter(config)
    elif router_type == "parallel":
        config = ParallelConfig(**resolved)
        return ParallelRouter(config)
    else:
        raise ValueError(f"Unknown router type '{router_type}'. Use 'cascade' or 'parallel'.")


def build_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build InjectionGuard constructor kwargs from a config dict.

    This returns a dict of kwargs that can be unpacked into the
    ``InjectionGuard`` constructor. It does NOT instantiate ``InjectionGuard``
    itself — that happens in ``InjectionGuard.from_config()``.

    Args:
        config: Parsed config dict (from YAML or programmatic construction).

    Returns:
        Dict of kwargs for ``InjectionGuard.__init__``.

    Example YAML structure::

        classifiers:
          - type: regex
          - type: anthropic
            model: claude-sonnet-4-20250514
            weight: 2.0
          - type: gemini
            model: gemini-2.0-flash
            project: ${GCP_PROJECT_ID}
            region: ${GCP_REGION}
          - type: openai
            model: gpt-4o
            weight: 1.5

        router:
          type: cascade
          timeout_ms: 500
          fast_confidence: 0.85

        thresholds:
          block: 0.85
          flag: 0.50

        aggregator: weighted_average

        preprocessor:
          gliner_model: urchade/gliner_base
          preprocessor_block_threshold: 0.9
    """
    kwargs: dict[str, Any] = {}

    # Classifiers
    clf_configs = config.get("classifiers", [])
    classifiers = [_build_classifier(dict(c)) for c in clf_configs]
    kwargs["classifiers"] = classifiers

    # Router
    router_cfg = config.get("router", {"type": "cascade"})
    kwargs["router"] = _build_router(dict(router_cfg))

    # Thresholds
    if "thresholds" in config:
        kwargs["thresholds"] = config["thresholds"]

    # Aggregator
    if "aggregator" in config:
        kwargs["aggregator"] = config["aggregator"]

    # Preprocessor settings
    prep = config.get("preprocessor", {})
    if "gliner_model" in prep:
        kwargs["gliner_model"] = prep["gliner_model"]
    if "gliner_device" in prep:
        kwargs["gliner_device"] = prep["gliner_device"]
    if "preprocessor_block_threshold" in prep:
        kwargs["preprocessor_block_threshold"] = prep["preprocessor_block_threshold"]

    # Meta-classifier
    if "meta_classifier_path" in config:
        kwargs["meta_classifier_path"] = config["meta_classifier_path"]

    # Dotenv
    if "dotenv_path" in config:
        kwargs["dotenv_path"] = config["dotenv_path"]

    return kwargs
