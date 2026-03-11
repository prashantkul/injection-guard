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
    _client: object = field(default=None, init=False, repr=False)

    def _get_client(self):
        """Lazily create and cache the Model Armor async client."""
        if self._client is not None:
            return self._client
        from google.cloud import modelarmor_v1  # type: ignore[import-untyped]
        from google.api_core.client_options import ClientOptions  # type: ignore[import-untyped]
        endpoint = f"modelarmor.{self.location}.rep.googleapis.com"
        self._client = modelarmor_v1.ModelArmorAsyncClient(
            client_options=ClientOptions(api_endpoint=endpoint),
        )
        return self._client

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
            client = self._get_client()

            template_name = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/templates/{self.template_id}"
            )

            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=template_name,
                user_prompt_data=modelarmor_v1.DataItem(
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
        if not filter_results:
            return result

        # SDK v0.4+ returns filter_results as a map (dict-like).
        # Support both dict-style .get() and older attribute access.
        def _get_filter(key: str) -> object | None:
            if hasattr(filter_results, "get"):
                return filter_results.get(key)
            return getattr(filter_results, key, None)

        # Confidence level enum → name mapping
        _CONF_NAMES = {0: "UNSPECIFIED", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}

        # --- Prompt injection & jailbreak ---
        pi_filter = _get_filter("pi_and_jailbreak")
        if pi_filter is not None:
            pi_result = getattr(pi_filter, "pi_and_jailbreak_filter_result", pi_filter)
            match_state = getattr(pi_result, "match_state", None)
            confidence = getattr(pi_result, "confidence_level", None)

            # Resolve enum to string name
            if isinstance(confidence, int):
                confidence_name = _CONF_NAMES.get(confidence, str(confidence))
            elif hasattr(confidence, "name"):
                confidence_name = confidence.name
            else:
                confidence_name = str(confidence)

            # match_state == 2 means MATCH_FOUND in the proto enum
            is_match = (match_state == 2) if isinstance(match_state, int) else (
                "MATCH_FOUND" in str(match_state)
            )

            if is_match and confidence_name in allowed_levels:
                result.match_found = True
                result.pi_and_jailbreak = True
                result.confidence_level = confidence_name

        # --- Malicious URIs ---
        url_filter = _get_filter("malicious_uris")
        if url_filter is not None:
            url_result = getattr(url_filter, "malicious_uri_filter_result", url_filter)
            uris = getattr(url_result, "malicious_uris", []) or []
            if uris:
                result.match_found = True
                result.malicious_urls = list(uris)

        # --- Responsible AI ---
        rai_filter = _get_filter("rai")
        if rai_filter is not None:
            rai_dict: dict = {}
            try:
                rai_dict = type(rai_filter).to_dict(rai_filter)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            if rai_dict:
                result.rai_findings = rai_dict

        return result
