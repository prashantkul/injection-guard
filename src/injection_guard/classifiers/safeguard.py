"""OpenAI gpt-oss-safeguard classifier for policy-based safety evaluation.

Uses gpt-oss-safeguard (20B or 120B) with a custom safety policy as the
system prompt. The model reasons about whether input violates the policy and
returns a structured verdict with category codes and confidence levels.

Safeguard is a **Stage 1 safety policy signal provider** — it evaluates
content against any structured policy (PI/JB, spam, content safety,
compliance, etc.) and contributes category-level signals to the SignalVector
for Stage 2 frontier classifiers. See docs/safeguard-policy.md for policy
examples and custom policy authoring.

Runs on Ollama, vLLM, or any OpenAI-compatible server.
Requires: ``pip install openai``
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from injection_guard.types import ClassifierResult, SignalVector
from injection_guard.classifiers.prompts import format_signals_context

__all__ = ["SafeguardClassifier"]

_DEFAULT_MODEL = "gpt-oss-safeguard:20b"
_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# Default safety policy fed as system prompt to gpt-oss-safeguard.
# This is a general-purpose safety policy — NOT a PI/JB classifier.
# DeBERTa and Model Armor handle PI/JB detection; Safeguard provides
# safety policy category signals to enrich the SignalVector for Stage 2.
#
# Override this with any custom policy (spam, content safety, compliance)
# by setting `self.policy` on the instance or passing a custom
# system_prompt. See docs/safeguard-policy.md for examples.
_DEFAULT_SAFETY_POLICY = """\
You are a safety policy evaluator. Evaluate the user content against the \
following policy and respond with ONLY a JSON object.

## Safety Policy

### P1: Violence & Threats
Content that threatens, promotes, glorifies, or incites violence against \
individuals or groups. Includes direct threats ("I will hurt you"), \
glorification of violent acts, or detailed instructions for carrying out \
violence.

### P2: Hate Speech & Discrimination
Content that attacks, demeans, or discriminates against individuals or groups \
based on protected characteristics: race, ethnicity, religion, gender, sexual \
orientation, disability, or nationality. Includes slurs, dehumanization, and \
calls for exclusion or harm.

### P3: Self-Harm & Suicide
Content that promotes, encourages, or provides instructions for self-harm or \
suicide. Includes glorification, methods, or pressure to engage in self-harm. \
Excludes educational or support-seeking content.

### P4: Sexual Content & Exploitation
Explicit sexual content, especially involving minors. Includes non-consensual \
sexual scenarios, sexual exploitation, grooming behavior, or sexualized \
content involving anyone under 18.

### P5: Dangerous Activities & Illegal Instructions
Content that provides actionable instructions for creating weapons, \
synthesizing drugs, making explosives, or other activities that pose serious \
physical harm. Includes step-by-step guides, precursor sourcing, and \
operational details.

### P6: Fraud, Deception & Social Engineering
Content designed to deceive, defraud, or manipulate. Includes phishing \
attempts, impersonation, scam scripts, fake identity generation, or social \
engineering tactics to extract sensitive information or money.

## Response Format

Respond with ONLY this JSON — no other text:
{{"violation": <0 or 1>, "categories": ["<P1-P6 codes>"], "confidence": "<low|medium|high>", "reasoning": "<brief explanation>"}}

- violation: 1 if ANY policy is violated, 0 if content is benign
- categories: list of violated policy codes (empty if benign)
- confidence: your confidence level
- reasoning: one-sentence explanation
"""


@dataclass
class SafeguardClassifier:
    """Stage 1 safety policy signal provider using gpt-oss-safeguard.

    Evaluates content against a configurable safety policy and returns
    structured verdicts with category codes and confidence levels. These
    signals enrich the SignalVector for Stage 2 frontier classifiers.

    Not a PI/JB classifier — DeBERTa and Model Armor handle that. Safeguard
    provides policy-level category signals (instruction override, role
    hijacking, prompt extraction, etc.) and supports any custom policy
    (spam, content safety, compliance).

    Works with Ollama, vLLM, or any OpenAI-compatible server hosting
    gpt-oss-safeguard.

    Attributes:
        model: Model name as known by the server.
        base_url: Server URL for OpenAI-compatible API.
        weight: Weight used by the ensemble aggregator.
        reasoning_effort: Safeguard reasoning depth ("low", "medium", "high").
        policy: Custom safety policy string. Defaults to built-in policy.
    """

    model: str = _DEFAULT_MODEL
    base_url: str = ""
    weight: float = 1.5
    latency_tier: str = "medium"
    reasoning_effort: str = "medium"
    api_key: str = "not-needed"
    policy: str = _DEFAULT_SAFETY_POLICY
    client: Any = field(default=None, repr=False)
    _name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get(
                "SAFEGUARD_BASE_URL",
                os.environ.get("LOCAL_LLM_BASE_URL", _DEFAULT_BASE_URL),
            )
        model_short = self.model.split(":")[0] if ":" in self.model else self.model
        self._name = f"safeguard-{model_short}"

    @property
    def name(self) -> str:  # noqa: D401
        """Classifier identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    async def _get_client(self) -> Any:
        """Return an async OpenAI-compatible client, creating one if needed."""
        if self.client is not None:
            return self.client
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for SafeguardClassifier. "
                "Install it with: pip install openai"
            ) from exc

        self.client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return self.client

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult:
        """Evaluate *prompt* against the safety policy using gpt-oss-safeguard.

        Returns policy violation signals (category codes, confidence, reasoning)
        for SignalVector enrichment. Never raises — returns a degraded result
        on any error.
        """
        try:
            client = await self._get_client()

            # Build user message with optional preprocessor signals
            user_content = prompt
            signals_ctx = format_signals_context(signals)
            if signals_ctx:
                user_content += signals_ctx

            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.policy},
                    {"role": "user", "content": user_content},
                ],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            raw_text = response.choices[0].message.content or ""
            data = _parse_safeguard_response(raw_text)

            violation = data.get("violation", 0)
            categories = data.get("categories", [])
            confidence_str = data.get("confidence", "medium")
            reasoning = data.get("reasoning", "")

            # Map to our score system
            confidence_map = {"low": 0.4, "medium": 0.7, "high": 0.95}
            confidence = confidence_map.get(confidence_str, 0.5)

            if violation:
                score = 0.7 + (confidence * 0.3)  # 0.82 - 0.985
            else:
                score = 0.1 * (1 - confidence)  # 0.005 - 0.06

            score = max(0.0, min(1.0, score))
            label = "injection" if violation else "benign"

            return ClassifierResult(
                score=score,
                label=label,
                confidence=confidence,
                reasoning=reasoning if reasoning else None,
                latency_ms=elapsed_ms,
                metadata={
                    "raw_response": raw_text,
                    "violation": violation,
                    "categories": categories,
                    "safeguard_confidence": confidence_str,
                    "base_url": self.base_url,
                },
            )

        except Exception as exc:  # noqa: BLE001
            return ClassifierResult(
                score=0.5,
                label="injection",
                confidence=0.0,
                metadata={"error": str(exc), "base_url": self.base_url},
            )


def _parse_safeguard_response(text: str) -> dict[str, Any]:
    """Parse gpt-oss-safeguard response, handling reasoning + output channels.

    The model may return reasoning text before the JSON, or wrap JSON in
    markdown fences. This extracts the first valid JSON object.
    """
    # Try to find JSON in the text
    # Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
    stripped = re.sub(r"\s*```$", "", stripped)

    # Try direct parse first
    try:
        return json.loads(stripped)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        pass

    # Find JSON object in mixed text (reasoning + output)
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            pass

    return {"violation": 1, "confidence": "low", "reasoning": "Failed to parse response"}
