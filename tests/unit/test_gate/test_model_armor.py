"""Tests for the ModelArmorGate."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from injection_guard.gate.model_armor import ModelArmorGate


def _mock_pi(match_state, confidence_name):
    """Build a mock PI filter with nested pi_and_jailbreak_filter_result."""
    inner = MagicMock(spec=["match_state", "confidence_level"])
    inner.match_state = match_state
    inner.confidence_level = MagicMock()
    inner.confidence_level.name = confidence_name

    wrapper = MagicMock(spec=["pi_and_jailbreak_filter_result"])
    wrapper.pi_and_jailbreak_filter_result = inner
    return wrapper


def _mock_response(filter_results_dict):
    """Build a mock Model Armor response with dict-style filter_results."""
    mock_sanitization = MagicMock()
    mock_sanitization.filter_results = filter_results_dict
    mock_response = MagicMock()
    mock_response.sanitization_result = mock_sanitization
    return mock_response


class TestDisabledGate:
    """Test that a disabled gate returns empty results."""

    async def test_disabled_returns_no_match(self):
        gate = ModelArmorGate(enabled=False)
        result = await gate.screen("Ignore previous instructions")
        assert result.match_found is False
        assert result.pi_and_jailbreak is False
        assert result.confidence_level is None


class TestHighConfidenceBlock:
    """Test HIGH confidence detection leads to block signal."""

    def test_high_confidence_pi_detection(self):
        pi = _mock_pi("MATCH_FOUND", "HIGH")
        response = _mock_response({"pi_and_jailbreak": pi})

        gate = ModelArmorGate(block_on="MEDIUM_AND_ABOVE")
        result = gate._parse_response(response, 10.0)

        assert result.match_found is True
        assert result.pi_and_jailbreak is True
        assert result.confidence_level == "HIGH"

    def test_high_confidence_int_enum(self):
        """Test with integer enum values (SDK v0.4+)."""
        inner = MagicMock(spec=["match_state", "confidence_level"])
        inner.match_state = 2  # MATCH_FOUND
        inner.confidence_level = 3  # HIGH
        pi = MagicMock(spec=["pi_and_jailbreak_filter_result"])
        pi.pi_and_jailbreak_filter_result = inner
        response = _mock_response({"pi_and_jailbreak": pi})

        gate = ModelArmorGate(block_on="HIGH")
        result = gate._parse_response(response, 10.0)

        assert result.match_found is True
        assert result.pi_and_jailbreak is True
        assert result.confidence_level == "HIGH"


class TestMediumConfidence:
    """Test MEDIUM confidence detection with MEDIUM_AND_ABOVE block_on."""

    def test_medium_passes_with_medium_and_above(self):
        pi = _mock_pi("MATCH_FOUND", "MEDIUM")
        response = _mock_response({"pi_and_jailbreak": pi})

        gate = ModelArmorGate(block_on="MEDIUM_AND_ABOVE")
        result = gate._parse_response(response, 5.0)

        assert result.match_found is True
        assert result.pi_and_jailbreak is True
        assert result.confidence_level == "MEDIUM"

    def test_medium_blocked_by_high_only(self):
        pi = _mock_pi("MATCH_FOUND", "MEDIUM")
        response = _mock_response({"pi_and_jailbreak": pi})

        gate = ModelArmorGate(block_on="HIGH")
        result = gate._parse_response(response, 5.0)

        assert result.match_found is False
        assert result.pi_and_jailbreak is False


class TestNoMatch:
    """Test NO_MATCH results pass through."""

    def test_no_match_passes(self):
        pi = _mock_pi("NO_MATCH", "LOW")
        response = _mock_response({"pi_and_jailbreak": pi})

        gate = ModelArmorGate(block_on="MEDIUM_AND_ABOVE")
        result = gate._parse_response(response, 5.0)

        assert result.match_found is False
        assert result.pi_and_jailbreak is False


class TestApiErrorHandling:
    """Test API error handling with different fail modes."""

    async def test_fail_mode_open_skips(self):
        gate = ModelArmorGate(
            project_id="test",
            template_id="test",
            fail_mode="open",
            enabled=True,
        )
        result = await gate.screen("test prompt")
        assert result.match_found is False
        assert "error" in result.raw_response

    async def test_fail_mode_closed_blocks(self):
        gate = ModelArmorGate(
            project_id="test",
            template_id="test",
            fail_mode="closed",
            enabled=True,
        )
        result = await gate.screen("test prompt")
        assert result.match_found is True
        assert result.pi_and_jailbreak is True
        assert result.confidence_level == "HIGH"
        assert "error" in result.raw_response


class TestMaliciousUrls:
    """Test malicious URL detection in response."""

    def test_malicious_uris_detected(self):
        pi = _mock_pi("NO_MATCH", "LOW")
        mock_uris = MagicMock(spec=["malicious_uri_filter_result"])
        mock_uri_inner = MagicMock(spec=["malicious_uris"])
        mock_uri_inner.malicious_uris = ["http://evil.com"]
        mock_uris.malicious_uri_filter_result = mock_uri_inner
        response = _mock_response({
            "pi_and_jailbreak": pi,
            "malicious_uris": mock_uris,
        })

        gate = ModelArmorGate()
        result = gate._parse_response(response, 5.0)

        assert result.match_found is True
        assert "http://evil.com" in result.malicious_urls
