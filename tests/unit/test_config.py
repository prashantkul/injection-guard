"""Tests for the config module."""
from __future__ import annotations

import pytest

from injection_guard.config import load_config, build_from_config, _resolve_env


class TestResolveEnv:
    """Test environment variable resolution in config values."""

    def test_plain_string_unchanged(self):
        assert _resolve_env("hello") == "hello"

    def test_env_var_resolved(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "my-value")
        assert _resolve_env("${MY_VAR}") == "my-value"

    def test_env_var_with_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert _resolve_env("${MISSING_VAR:-fallback}") == "fallback"

    def test_env_var_missing_no_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert _resolve_env("${MISSING_VAR}") == ""

    def test_non_string_passthrough(self):
        assert _resolve_env(42) == 42
        assert _resolve_env(3.14) == 3.14
        assert _resolve_env(True) is True


class TestLoadConfig:
    """Test YAML config loading."""

    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "classifiers:\n"
            "  - type: regex\n"
            "thresholds:\n"
            "  block: 0.90\n"
            "  flag: 0.40\n"
        )
        config = load_config(str(config_file))
        assert config["classifiers"][0]["type"] == "regex"
        assert config["thresholds"]["block"] == 0.90

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestBuildFromConfig:
    """Test building InjectionGuard kwargs from config dicts."""

    def test_minimal_config_with_regex(self):
        config = {
            "classifiers": [{"type": "regex"}],
        }
        kwargs = build_from_config(config)
        assert len(kwargs["classifiers"]) == 1
        assert kwargs["classifiers"][0].name == "regex-prefilter"
        assert "router" in kwargs

    def test_config_with_thresholds(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "thresholds": {"block": 0.90, "flag": 0.40},
        }
        kwargs = build_from_config(config)
        assert kwargs["thresholds"]["block"] == 0.90

    def test_config_with_aggregator(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "aggregator": "voting",
        }
        kwargs = build_from_config(config)
        assert kwargs["aggregator"] == "voting"

    def test_config_with_preprocessor_settings(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "preprocessor": {
                "gliner_model": "custom/model",
                "preprocessor_block_threshold": 0.8,
            },
        }
        kwargs = build_from_config(config)
        assert kwargs["gliner_model"] == "custom/model"
        assert kwargs["preprocessor_block_threshold"] == 0.8

    def test_config_with_cascade_router(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "router": {
                "type": "cascade",
                "timeout_ms": 1000,
                "fast_confidence": 0.90,
            },
        }
        kwargs = build_from_config(config)
        assert kwargs["router"] is not None

    def test_config_with_parallel_router(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "router": {
                "type": "parallel",
                "quorum": 3,
            },
        }
        kwargs = build_from_config(config)
        assert kwargs["router"] is not None

    def test_config_with_category_quorum(self):
        config = {
            "classifiers": [
                {"type": "regex", "category": "local"},
                {"type": "openai", "model": "gpt-4o", "category": "api"},
            ],
            "router": {
                "type": "parallel",
                "timeout_ms": 5000,
                "category_quorum": {"local": 1, "api": 1},
            },
        }
        kwargs = build_from_config(config)
        router = kwargs["router"]
        assert router._config.category_quorum == {"local": 1, "api": 1}
        assert "regex-prefilter" in router._config.classifier_categories
        assert router._config.classifier_categories["regex-prefilter"] == "local"

    def test_category_not_injected_for_cascade(self):
        config = {
            "classifiers": [
                {"type": "regex", "category": "local"},
            ],
            "router": {"type": "cascade"},
        }
        kwargs = build_from_config(config)
        # Cascade router doesn't use categories — should still build fine
        assert kwargs["router"] is not None

    def test_unknown_classifier_type_raises(self):
        config = {
            "classifiers": [{"type": "unknown_thing"}],
        }
        with pytest.raises(ValueError, match="Unknown classifier type"):
            build_from_config(config)

    def test_unknown_router_type_raises(self):
        config = {
            "classifiers": [{"type": "regex"}],
            "router": {"type": "unknown_router"},
        }
        with pytest.raises(ValueError, match="Unknown router type"):
            build_from_config(config)

    def test_env_var_resolution_in_classifier(self, monkeypatch):
        monkeypatch.setenv("MY_MODEL", "gemini-2.0-pro")
        config = {
            "classifiers": [
                {"type": "gemini", "model": "${MY_MODEL}", "project": "test-proj"},
            ],
        }
        kwargs = build_from_config(config)
        clf = kwargs["classifiers"][0]
        assert clf.model == "gemini-2.0-pro"

    def test_multiple_classifiers(self):
        config = {
            "classifiers": [
                {"type": "regex"},
                {"type": "openai", "model": "gpt-4o", "weight": 1.5},
            ],
        }
        kwargs = build_from_config(config)
        assert len(kwargs["classifiers"]) == 2


class TestFromConfigIntegration:
    """Test InjectionGuard.from_config end-to-end."""

    def test_from_config_dict(self):
        from injection_guard.guard import InjectionGuard

        config = {
            "classifiers": [{"type": "regex"}],
            "thresholds": {"block": 0.85, "flag": 0.50},
        }
        guard = InjectionGuard.from_config(config)
        assert guard is not None
        assert len(guard._classifiers) == 1

    def test_from_config_yaml_file(self, tmp_path):
        from injection_guard.guard import InjectionGuard

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "classifiers:\n"
            "  - type: regex\n"
            "  - type: regex\n"
            "    weight: 0.3\n"
            "thresholds:\n"
            "  block: 0.90\n"
            "  flag: 0.40\n"
            "aggregator: weighted_average\n"
        )
        guard = InjectionGuard.from_config(str(config_file))
        assert len(guard._classifiers) == 2

    async def test_from_config_classify(self):
        from injection_guard.guard import InjectionGuard
        from injection_guard.types import Action

        config = {
            "classifiers": [{"type": "regex"}],
            "thresholds": {"block": 0.85, "flag": 0.50},
            "router": {"type": "cascade"},
        }
        guard = InjectionGuard.from_config(config)
        decision = await guard.classify("What is the capital of France?")
        assert decision.action == Action.ALLOW
