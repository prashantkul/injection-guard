"""Batch API adapters for large-scale evaluation.

Supports OpenAI Batch API, Anthropic Message Batches, and Gemini via
Vertex AI BatchPredictionJob. Each adapter submits classification prompts
in bulk, polls for completion, and parses results into ClassifierResults.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
from typing import Any

from injection_guard.types import ClassifierResult, SignalVector
from injection_guard.classifiers.prompts import (
    CLASSIFICATION_PROMPT,
    make_delimited_prompt,
    format_signals_context,
    extract_json,
    validate_result,
)

__all__ = [
    "OpenAIBatchAdapter",
    "AnthropicBatchAdapter",
    "GeminiBatchAdapter",
]


def _build_messages(prompt: str, signals: SignalVector | None = None) -> dict:
    """Build the classification prompt for a single sample."""
    delimited, _nonce = make_delimited_prompt(prompt)
    signals_ctx = format_signals_context(signals)
    full_prompt = CLASSIFICATION_PROMPT.format(
        delimited_prompt=delimited, signals_context=signals_ctx,
    )
    return full_prompt


class OpenAIBatchAdapter:
    """Adapter for the OpenAI Batch API.

    Submits a JSONL file of chat completion requests, polls for completion,
    and parses results. Batch API gives 50% cost reduction.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Return a synchronous OpenAI client."""
        import openai
        return openai.OpenAI(
            api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=120.0,
        )

    async def run_batch(
        self,
        prompts: list[str],
        signals_list: list[SignalVector | None] | None = None,
        poll_interval_s: float = 10.0,
        timeout_s: float = 3600.0,
    ) -> list[ClassifierResult]:
        """Submit prompts as a batch, wait for completion, return results.

        Args:
            prompts: List of user prompts to classify.
            signals_list: Optional per-prompt signal vectors.
            poll_interval_s: Seconds between status polls.
            timeout_s: Maximum wait time.

        Returns:
            List of ClassifierResult, one per prompt, in input order.
        """
        client = self._get_client()
        signals = signals_list or [None] * len(prompts)

        # Build JSONL input
        lines = []
        for i, (prompt, sig) in enumerate(zip(prompts, signals)):
            full_prompt = _build_messages(prompt, sig)
            request = {
                "custom_id": f"eval-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "max_completion_tokens": 256,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "You are a prompt-injection classifier."},
                        {"role": "user", "content": full_prompt},
                    ],
                },
            }
            lines.append(json.dumps(request))

        jsonl_content = "\n".join(lines)

        # Upload input file
        input_file = await asyncio.to_thread(
            client.files.create,
            file=io.BytesIO(jsonl_content.encode()),
            purpose="batch",
        )

        # Create batch
        batch = await asyncio.to_thread(
            client.batches.create,
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(f"    OpenAI batch created: {batch.id}", flush=True)

        # Poll for completion (with retry on transient HTTP errors)
        start = time.monotonic()
        while True:
            try:
                status = await asyncio.to_thread(client.batches.retrieve, batch.id)
            except Exception as poll_err:
                if time.monotonic() - start > timeout_s:
                    raise
                print(f"    OpenAI poll error (retrying): {poll_err}", flush=True)
                await asyncio.sleep(poll_interval_s)
                continue
            if status.status == "completed":
                break
            if status.status in ("failed", "cancelled", "expired"):
                raise RuntimeError(f"OpenAI batch {batch.id} {status.status}")
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"OpenAI batch {batch.id} timed out")
            completed = status.request_counts.completed if status.request_counts else 0
            total = status.request_counts.total if status.request_counts else len(prompts)
            print(f"    OpenAI batch: {completed}/{total} ({status.status})", flush=True)
            await asyncio.sleep(poll_interval_s)

        # Download results
        output_file_id = status.output_file_id
        if not output_file_id:
            # All requests may have errored — check error file
            error_file_id = status.error_file_id
            if error_file_id:
                err_content = await asyncio.to_thread(client.files.content, error_file_id)
                print(f"    OpenAI batch errors: {err_content.text[:500]}", flush=True)
            raise RuntimeError(
                f"OpenAI batch {batch.id} completed with no output file. "
                f"Counts: {status.request_counts}"
            )
        content = await asyncio.to_thread(client.files.content, output_file_id)
        result_lines = content.text.strip().split("\n")

        # Parse results, keyed by custom_id
        results_map: dict[int, ClassifierResult] = {}
        for line in result_lines:
            record = json.loads(line)
            idx = int(record["custom_id"].split("-")[1])
            body = record.get("response", {}).get("body", {})
            try:
                raw_text = body["choices"][0]["message"]["content"]
                data = extract_json(raw_text)
                cr = validate_result(data)
                cr.metadata["raw_response"] = raw_text
            except Exception as exc:
                cr = ClassifierResult(
                    score=0.5, label="injection", confidence=0.0,
                    metadata={"error": str(exc)},
                )
            results_map[idx] = cr

        return [results_map.get(i, ClassifierResult(
            score=0.5, label="injection", confidence=0.0,
            metadata={"error": "missing from batch output"},
        )) for i in range(len(prompts))]


class AnthropicBatchAdapter:
    """Adapter for the Anthropic Message Batches API.

    Submits classification requests as a message batch, polls for
    completion, and parses results. Batch API gives 50% cost reduction.

    Args:
        model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY``.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Return a synchronous Anthropic client."""
        import anthropic
        return anthropic.Anthropic(
            api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def run_batch(
        self,
        prompts: list[str],
        signals_list: list[SignalVector | None] | None = None,
        poll_interval_s: float = 10.0,
        timeout_s: float = 3600.0,
    ) -> list[ClassifierResult]:
        """Submit prompts as a message batch, wait, return results.

        Args:
            prompts: List of user prompts to classify.
            signals_list: Optional per-prompt signal vectors.
            poll_interval_s: Seconds between status polls.
            timeout_s: Maximum wait time.

        Returns:
            List of ClassifierResult, one per prompt, in input order.
        """
        client = self._get_client()
        signals = signals_list or [None] * len(prompts)

        # Build batch requests
        requests = []
        for i, (prompt, sig) in enumerate(zip(prompts, signals)):
            full_prompt = _build_messages(prompt, sig)
            requests.append({
                "custom_id": f"eval-{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": 256,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": full_prompt}],
                },
            })

        # Create batch
        batch = await asyncio.to_thread(
            client.messages.batches.create,
            requests=requests,
        )

        print(f"    Anthropic batch created: {batch.id}", flush=True)

        # Poll for completion
        start = time.monotonic()
        while True:
            status = await asyncio.to_thread(
                client.messages.batches.retrieve, batch.id,
            )
            if status.processing_status == "ended":
                break
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"Anthropic batch {batch.id} timed out")
            counts = status.request_counts
            done = (counts.succeeded or 0) + (counts.errored or 0)
            total = counts.processing + done
            print(f"    Anthropic batch: {done}/{total} ({status.processing_status})", flush=True)
            await asyncio.sleep(poll_interval_s)

        # Collect results
        results_map: dict[int, ClassifierResult] = {}
        for entry in client.messages.batches.results(batch.id):
            idx = int(entry.custom_id.split("-")[1])
            if entry.result.type == "succeeded":
                try:
                    raw_text = entry.result.message.content[0].text
                    data = extract_json(raw_text)
                    cr = validate_result(data)
                    cr.metadata["raw_response"] = raw_text
                except Exception as exc:
                    cr = ClassifierResult(
                        score=0.5, label="injection", confidence=0.0,
                        metadata={"error": str(exc)},
                    )
            else:
                cr = ClassifierResult(
                    score=0.5, label="injection", confidence=0.0,
                    metadata={"error": f"batch entry {entry.result.type}"},
                )
            results_map[idx] = cr

        return [results_map.get(i, ClassifierResult(
            score=0.5, label="injection", confidence=0.0,
            metadata={"error": "missing from batch output"},
        )) for i in range(len(prompts))]


class GeminiBatchAdapter:
    """Adapter for the Gemini Batch API via google-genai.

    Uses the native Gemini batch endpoint for 50% cost reduction.
    Submits inline requests, polls for completion, and parses results.

    Args:
        model: Gemini model identifier.
        api_key: Gemini API key. Falls back to ``GOOGLE_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Return a google-genai client using the Developer API (not Vertex)."""
        from google import genai
        return genai.Client(
            api_key=self._api_key or os.environ.get("GOOGLE_API_KEY"),
            vertexai=False,
        )

    async def run_batch(
        self,
        prompts: list[str],
        signals_list: list[SignalVector | None] | None = None,
        poll_interval_s: float = 30.0,
        timeout_s: float = 3600.0,
    ) -> list[ClassifierResult]:
        """Submit prompts as a Gemini batch, wait for completion, return results.

        Args:
            prompts: List of user prompts to classify.
            signals_list: Optional per-prompt signal vectors.
            poll_interval_s: Seconds between status polls.
            timeout_s: Maximum wait time.

        Returns:
            List of ClassifierResult, one per prompt, in input order.
        """
        client = self._get_client()
        signals = signals_list or [None] * len(prompts)

        # Build inline requests using typed google-genai objects
        from google.genai import types as genai_types
        inline_requests = []
        for i, (prompt, sig) in enumerate(zip(prompts, signals)):
            full_prompt = _build_messages(prompt, sig)
            inline_requests.append(genai_types.InlinedRequest(
                contents=full_prompt,
                metadata={"key": f"eval-{i}"},
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=256,
                ),
            ))

        # Create batch job
        batch_job = await asyncio.to_thread(
            client.batches.create,
            model=self.model,
            src=inline_requests,
            config={"display_name": "injection-guard-eval"},
        )

        job_name = batch_job.name
        print(f"    Gemini batch created: {job_name}", flush=True)

        # Poll for completion
        start = time.monotonic()
        while True:
            batch_job = await asyncio.to_thread(
                client.batches.get, name=job_name,
            )
            state = batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)
            if state in ("JOB_STATE_SUCCEEDED", "SUCCEEDED"):
                break
            if state in ("JOB_STATE_FAILED", "FAILED", "JOB_STATE_CANCELLED", "CANCELLED"):
                raise RuntimeError(f"Gemini batch {job_name} {state}")
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"Gemini batch {job_name} timed out")
            print(f"    Gemini batch: {state}", flush=True)
            await asyncio.sleep(poll_interval_s)

        # Parse results — responses come back in order (positional index)
        results_map: dict[int, ClassifierResult] = {}
        dest = batch_job.dest

        if dest and hasattr(dest, "inlined_responses") and dest.inlined_responses:
            for idx, entry in enumerate(dest.inlined_responses):
                try:
                    resp = entry.response
                    # Try candidates path first, then .text
                    if hasattr(resp, "candidates") and resp.candidates:
                        raw_text = resp.candidates[0].content.parts[0].text
                    elif hasattr(resp, "text") and resp.text:
                        raw_text = resp.text
                    else:
                        raise ValueError(f"No text in response: {resp}")
                    data = extract_json(raw_text)
                    cr = validate_result(data)
                    cr.metadata["raw_response"] = raw_text
                except Exception as exc:
                    cr = ClassifierResult(
                        score=0.5, label="injection", confidence=0.0,
                        metadata={"error": str(exc)},
                    )
                results_map[idx] = cr
        elif dest and hasattr(dest, "file_name") and dest.file_name:
            # Results in output file
            content = await asyncio.to_thread(
                client.files.download, file=dest.file_name,
            )
            content_str = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            for idx, line in enumerate(content_str.splitlines()):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    raw_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    data = extract_json(raw_text)
                    cr = validate_result(data)
                    cr.metadata["raw_response"] = raw_text
                except Exception as exc:
                    cr = ClassifierResult(
                        score=0.5, label="injection", confidence=0.0,
                        metadata={"error": str(exc)},
                    )
                results_map[idx] = cr

        return [results_map.get(i, ClassifierResult(
            score=0.5, label="injection", confidence=0.0,
            metadata={"error": "missing from batch output"},
        )) for i in range(len(prompts))]
