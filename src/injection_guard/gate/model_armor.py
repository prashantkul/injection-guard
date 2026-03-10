"""Google Cloud Model Armor gate for pre-screening prompts."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from injection_guard.types import ModelArmorResult

__all__ = ["ModelArmorGate"]

# Confidence levels ordered from lowest to highest severity.
_CONFIDENCE_LEVELS = ("LOW", "MEDIUM", "HIGH")

_BLOCK_ON_THRESHOLDS: dict[str, set[str]] = {
    "HIGH": {"HIGH"},
    "MEDIUM_AND_ABOVE": {"MEDIUM", "HIGH"},
    "LOW_AND_ABOVE": {"LOW", "MEDIUM", "HIGH"},
}


@dataclass
class ModelArmorGate:
    """Pre-classifier gate that uses Google Cloud Model Armor to screen prompts.

    Model Armor provides an additional layer of defence by detecting prompt
    injection, jailbreak, malicious URLs, and sensitive-data leakage before
    the prompt reaches the classifier ensemble.

    Attributes:
        project_id: GCP project identifier.
        location: GCP region for the Model Armor endpoint.
        template_id: Model Armor template to evaluate against.
        block_on: Minimum confidence level that triggers a block.
        fail_mode: Behaviour when the API call fails.
            ``"open"`` lets the prompt through; ``"closed"`` blocks it.
        enabled: Whether the gate is active.
    """

    project_id: str = ""
    location: str = "us-central1"
    template_id: str = ""
    block_on: Literal["HIGH", "MEDIUM_AND_ABOVE", "LOW_AND_ABOVE"] = (
        "MEDIUM_AND_ABOVE"
    )
    fail_mode: Literal["open", "closed"] = "closed"
    enabled: bool = True

    async def screen(self, prompt: str) -> ModelArmorResult:
        """Screen a prompt through Google Cloud Model Armor.

        Args:
            prompt: The raw user prompt to evaluate.

        Returns:
            A ``ModelArmorResult`` describing Model Armor's findings.  When the
            gate is disabled an empty (no-match) result is returned immediately.
        """
        if not self.enabled:
            return ModelArmorResult()

        start = time.perf_counter()

        try:
            from google.cloud import modelarmor_v1  # type: ignore[import-untyped]
        except ImportError:
            return self._handle_error(
                start,
                RuntimeError(
                    "google-cloud-modelarmor is not installed. "
                    "Install it with: pip install google-cloud-modelarmor"
                ),
            )

        try:
            client = modelarmor_v1.ModelArmorAsyncClient()

            template_name = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/templates/{self.template_id}"
            )

            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=template_name,
                user_prompt_data=modelarmor_v1.UserPromptData(
                    text=prompt,
                ),
            )

            response = await client.sanitize_user_prompt(request=request)

            latency_ms = (time.perf_counter() - start) * 1000
            return self._parse_response(response, latency_ms)

        except Exception as exc:  # noqa: BLE001
            return self._handle_error(start, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_error(
        self, start: float, exc: Exception
    ) -> ModelArmorResult:
        """Return a result consistent with the configured *fail_mode*.

        Args:
            start: Monotonic time when the request started.
            exc: The exception that was raised.

        Returns:
            A ``ModelArmorResult`` that either skips (open) or blocks
            (closed) depending on the configured fail mode.
        """
        latency_ms = (time.perf_counter() - start) * 1000

        if self.fail_mode == "open":
            return ModelArmorResult(
                match_found=False,
                latency_ms=latency_ms,
                raw_response={"error": str(exc)},
            )

        # fail_mode == "closed" — treat as worst-case detection.
        return ModelArmorResult(
            match_found=True,
            pi_and_jailbreak=True,
            confidence_level="HIGH",
            latency_ms=latency_ms,
            raw_response={"error": str(exc)},
        )

    def _parse_response(
        self, response: object, latency_ms: float
    ) -> ModelArmorResult:
        """Translate the Model Armor API response into a ``ModelArmorResult``.

        Args:
            response: The sanitization response object from the SDK.
            latency_ms: Pre-computed latency of the API call.

        Returns:
            A populated ``ModelArmorResult``.
        """
        raw: dict = {}
        try:
            raw = type(response).to_dict(response)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

        result = ModelArmorResult(latency_ms=latency_ms, raw_response=raw)
        allowed_levels = _BLOCK_ON_THRESHOLDS.get(self.block_on, {"HIGH"})

        sanitization = getattr(response, "sanitization_result", None)
        if sanitization is None:
            return result

        filter_results = getattr(sanitization, "filter_results", None)
        if filter_results is None:
            return result

        # --- Prompt injection & jailbreak ---
        pi_result = getattr(filter_results, "pi_and_jailbreak", None)
        if pi_result is not None:
            match_state = getattr(pi_result, "match_state", None)
            confidence = getattr(pi_result, "confidence_level", None)
            confidence_name = (
                confidence.name if hasattr(confidence, "name") else str(confidence)
            )

            if str(match_state) != "NO_MATCH" and confidence_name in allowed_levels:
                result.match_found = True
                result.pi_and_jailbreak = True
                result.confidence_level = confidence_name

        # --- Malicious URIs ---
        url_result = getattr(filter_results, "malicious_uris", None)
        if url_result is not None:
            uris = getattr(url_result, "malicious_uris", []) or []
            if uris:
                result.match_found = True
                result.malicious_urls = list(uris)

        # --- Sensitive data protection ---
        sdp_result = getattr(filter_results, "sdp", None)
        if sdp_result is not None:
            findings = getattr(sdp_result, "inspection_result", None)
            if findings:
                sdp_items = getattr(findings, "findings", []) or []
                result.sdp_findings = [
                    getattr(f, "info_type", str(f)) for f in sdp_items
                ]
                if result.sdp_findings:
                    result.match_found = True

        # --- Responsible AI ---
        rai_result = getattr(filter_results, "rai", None)
        if rai_result is not None:
            rai_dict: dict = {}
            try:
                rai_dict = type(rai_result).to_dict(rai_result)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            if rai_dict:
                result.rai_findings = rai_dict

        return result
