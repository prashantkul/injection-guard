"""Full ensemble integration test — runs all classifiers from config.yaml.

Uses real prompts from Qualifire (PI/JB) and ToxicChat (safety/toxicity)
via the shared dataset loader.

Requirements:
  - API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY in .env
  - GCP: GCP_PROJECT_ID for Model Armor and Gemini
  - DGX: Ollama at SAFEGUARD_BASE_URL (default 192.168.1.199:11434) for Safeguard
  - DGX: litguard at 192.168.1.199:8234 for HF DeBERTa
  - HF_TOKEN for gated Qualifire dataset
  - pip install datasets
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from injection_guard.guard import InjectionGuard
from injection_guard.reporting import print_decision, print_batch
from injection_guard.types import Action

try:
    from injection_guard.eval.dataset import (
        TestSample,
        load_qualifire,
        load_toxicchat,
        load_mixed,
    )
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def _check_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    """Check if a TCP connection can be established."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _check_ollama(host: str, port: int, timeout: float = 15.0) -> bool:
    """Check if Ollama can actually run a safeguard inference within timeout.

    Does a real (minimal) chat completion to verify end-to-end — model listing
    alone is not sufficient since cold-loading the model can exceed the router
    timeout.
    """
    import urllib.request
    import json

    try:
        url = f"http://{host}:{port}/v1/chat/completions"
        payload = json.dumps({
            "model": "gpt-oss-safeguard:20b",
            "messages": [
                {"role": "system", "content": "Reply OK."},
                {"role": "user", "content": "test"},
            ],
            "max_tokens": 5,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return "choices" in data
    except Exception:
        return False


def _has_env(key: str) -> bool:
    """Check if an environment variable is set and non-empty."""
    return bool(os.environ.get(key))


def _get_safeguard_url() -> tuple[str, int]:
    """Parse Safeguard base URL into (host, port)."""
    url = os.environ.get(
        "SAFEGUARD_BASE_URL",
        os.environ.get("LOCAL_LLM_BASE_URL", "http://192.168.1.199:11434/v1"),
    )
    parsed = urlparse(url)
    return parsed.hostname or "192.168.1.199", parsed.port or 11434


class PrereqResults:
    """Cached prerequisite check results (computed once at import time)."""

    has_config = CONFIG_PATH.exists()
    has_openai_key = _has_env("OPENAI_API_KEY")
    has_anthropic_key = _has_env("ANTHROPIC_API_KEY")
    has_gcp_project = _has_env("GCP_PROJECT_ID")
    has_hf_token = _has_env("HF_TOKEN")
    has_datasets_pkg = HAS_DATASETS

    _sg_host, _sg_port = _get_safeguard_url()
    has_safeguard = _check_ollama(_sg_host, _sg_port)
    has_litguard = _check_tcp("192.168.1.199", 8234)


prereqs = PrereqResults()


def _print_prereq_summary() -> None:
    """Print prerequisite checks as a Rich table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Integration Test Prerequisites", show_lines=False)
    table.add_column("Prerequisite", style="cyan")
    table.add_column("Status", justify="center")

    checks = [
        ("config.yaml", prereqs.has_config),
        ("OPENAI_API_KEY", prereqs.has_openai_key),
        ("ANTHROPIC_API_KEY", prereqs.has_anthropic_key),
        ("GCP_PROJECT_ID", prereqs.has_gcp_project),
        ("HF_TOKEN", prereqs.has_hf_token),
        ("datasets package", prereqs.has_datasets_pkg),
        (f"Safeguard model (Ollama @ {prereqs._sg_host}:{prereqs._sg_port})", prereqs.has_safeguard),
        ("litguard (DeBERTa @ 192.168.1.199:8234)", prereqs.has_litguard),
    ]
    for name, ok in checks:
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        table.add_row(name, status)

    console.print()
    console.print(table)
    console.print()


# Print prereq summary when module loads (visible in pytest -v output)
_print_prereq_summary()


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

skip_no_config = pytest.mark.skipif(
    not prereqs.has_config, reason="config.yaml not found",
)
skip_no_api_keys = pytest.mark.skipif(
    not (prereqs.has_openai_key and prereqs.has_anthropic_key),
    reason="Missing API keys (need OPENAI_API_KEY and ANTHROPIC_API_KEY)",
)
skip_no_datasets = pytest.mark.skipif(
    not prereqs.has_datasets_pkg, reason="datasets package not installed",
)
skip_no_hf_token = pytest.mark.skipif(
    not prereqs.has_hf_token, reason="HF_TOKEN not set (needed for Qualifire)",
)
skip_no_safeguard = pytest.mark.skipif(
    not prereqs.has_safeguard,
    reason=f"Safeguard not reachable at {prereqs._sg_host}:{prereqs._sg_port}",
)
skip_no_litguard = pytest.mark.skipif(
    not prereqs.has_litguard, reason="litguard not reachable at 192.168.1.199:8234",
)
skip_no_gcp = pytest.mark.skipif(
    not prereqs.has_gcp_project, reason="GCP_PROJECT_ID not set",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum samples for statistically meaningful results.
# Below this threshold, tests still run but print a warning.
MIN_SIGNIFICANT_SAMPLES = 30

# Default sample sizes per dataset
QUALIFIRE_N = 10
TOXICCHAT_N = 10
MIXED_N_PER_SOURCE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_dataset_summary(name: str, samples: list) -> None:
    """Print dataset setup info with class distribution and significance warning."""
    from collections import Counter

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    labels = Counter(s.label for s in samples)
    sources = Counter(s.source for s in samples)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Total samples", str(len(samples)))
    table.add_row("Sources", ", ".join(f"{k}: {v}" for k, v in sources.items()))
    table.add_row("Labels", ", ".join(f"{k}: {v}" for k, v in labels.items()))

    if len(samples) < MIN_SIGNIFICANT_SAMPLES:
        table.add_row(
            "[yellow]Warning[/yellow]",
            f"[yellow]{len(samples)} samples < {MIN_SIGNIFICANT_SAMPLES} minimum "
            f"for statistical significance[/yellow]",
        )

    console.print(Panel(table, title=f"Dataset: {name}", border_style="blue"))


def _print_test_summary(
    test_name: str,
    total: int,
    passed: int,
    failed_prompts: list[tuple[str, str, float]],
) -> None:
    """Print test result summary with pass/fail breakdown.

    Args:
        test_name: Name of the test.
        total: Total samples tested.
        passed: Number that passed the assertion.
        failed_prompts: List of (prompt_snippet, action, score) for failures.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    rate = passed / total if total > 0 else 0
    color = "green" if not failed_prompts else "red"

    lines = [f"[bold]Passed:[/bold] {passed}/{total} ({rate:.0%})"]

    if failed_prompts:
        lines.append(f"[bold red]Failed:[/bold red] {len(failed_prompts)}/{total}")
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Action", style="red")
        table.add_column("Score", justify="right")
        table.add_column("Prompt")
        for prompt, action, score in failed_prompts[:5]:
            table.add_row(action, f"{score:.3f}", prompt)
        if len(failed_prompts) > 5:
            table.add_row("", "", f"... and {len(failed_prompts) - 5} more")

        console.print(Panel(
            "\n".join(lines),
            title=f"Summary: {test_name}",
            border_style=color,
        ))
        console.print(table)
    else:
        console.print(Panel(
            "\n".join(lines),
            title=f"Summary: {test_name}",
            border_style=color,
        ))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def guard() -> InjectionGuard:
    """Load guard from config.yaml."""
    return InjectionGuard.from_config(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def qualifire_samples() -> list:
    """Balanced samples from Qualifire."""
    samples = load_qualifire(n=QUALIFIRE_N, seed=42)
    _print_dataset_summary("Qualifire (PI/JB)", samples)
    return samples


@pytest.fixture(scope="module")
def toxicchat_samples() -> list:
    """Balanced samples from ToxicChat."""
    samples = load_toxicchat(n=TOXICCHAT_N, seed=42)
    _print_dataset_summary("ToxicChat (Safety)", samples)
    return samples


@pytest.fixture(scope="module")
def mixed_samples() -> list:
    """Mixed samples from both datasets."""
    samples = load_mixed(n_per_source=MIXED_N_PER_SOURCE, seed=42)
    _print_dataset_summary("Mixed (Qualifire + ToxicChat)", samples)
    return samples


# ---------------------------------------------------------------------------
# Tests: Qualifire PI/JB
# ---------------------------------------------------------------------------

@skip_no_config
@skip_no_api_keys
@skip_no_datasets
@skip_no_hf_token
class TestQualifirePIJB:
    """PI/JB detection using real Qualifire prompts."""

    async def test_injections_detected(
        self, guard: InjectionGuard, qualifire_samples: list[TestSample],
    ):
        """Real PI/JB prompts from Qualifire should be BLOCK or FLAG."""
        injections = [s for s in qualifire_samples if s.label == "injection"]
        assert len(injections) > 0, "No injection samples loaded"

        passed = 0
        failures: list[tuple[str, str, float]] = []

        for sample in injections:
            decision = await guard.classify(sample.prompt)
            print_decision(decision, show_prompt=True)
            if decision.action in (Action.BLOCK, Action.FLAG):
                passed += 1
            else:
                failures.append((
                    sample.prompt[:80], decision.action.value, decision.ensemble_score,
                ))

        _print_test_summary("Qualifire injections → BLOCK/FLAG", len(injections), passed, failures)

        assert not failures, (
            f"{len(failures)}/{len(injections)} injection prompts were not BLOCK/FLAG"
        )

    async def test_benign_not_blocked(
        self, guard: InjectionGuard, qualifire_samples: list[TestSample],
    ):
        """Benign prompts should never be BLOCK. FLAG is tolerable (FP)."""
        benign = [s for s in qualifire_samples if s.label == "benign"]
        assert len(benign) > 0, "No benign samples loaded"

        passed = 0
        blocked: list[tuple[str, str, float]] = []
        flagged = 0

        for sample in benign:
            decision = await guard.classify(sample.prompt)
            print_decision(decision, show_prompt=True)
            if decision.action == Action.BLOCK:
                blocked.append((
                    sample.prompt[:80], decision.action.value, decision.ensemble_score,
                ))
            else:
                passed += 1
                if decision.action == Action.FLAG:
                    flagged += 1

        _print_test_summary("Qualifire benign → not BLOCK", len(benign), passed, blocked)
        if flagged:
            from rich.console import Console
            Console().print(
                f"  [yellow]Warning:[/yellow] {flagged}/{len(benign)} benign prompts "
                f"were FLAG'd (false positives)"
            )

        assert not blocked, (
            f"{len(blocked)}/{len(benign)} benign prompts were BLOCKED"
        )


# ---------------------------------------------------------------------------
# Tests: ToxicChat safety
# ---------------------------------------------------------------------------

@skip_no_config
@skip_no_api_keys
@skip_no_datasets
@skip_no_safeguard
class TestToxicChatSafety:
    """Safety detection using real ToxicChat prompts.

    Requires Safeguard (Ollama) to be reachable — without it, the
    ensemble has no safety policy classifier and toxic-but-not-injection
    prompts will pass through.
    """

    async def test_toxic_detected(
        self, guard: InjectionGuard, toxicchat_samples: list[TestSample],
    ):
        """Real toxic prompts from ToxicChat should be BLOCK or FLAG."""
        toxic = [s for s in toxicchat_samples if s.label == "toxic"]
        assert len(toxic) > 0, "No toxic samples loaded"

        passed = 0
        failures: list[tuple[str, str, float]] = []

        for sample in toxic:
            decision = await guard.classify(sample.prompt)
            print_decision(decision, show_prompt=True)
            if decision.action in (Action.BLOCK, Action.FLAG):
                passed += 1
            else:
                failures.append((
                    sample.prompt[:80], decision.action.value, decision.ensemble_score,
                ))

        _print_test_summary("ToxicChat toxic → BLOCK/FLAG", len(toxic), passed, failures)

        assert not failures, (
            f"{len(failures)}/{len(toxic)} toxic prompts were not BLOCK/FLAG"
        )

    async def test_safe_allowed(
        self, guard: InjectionGuard, toxicchat_samples: list[TestSample],
    ):
        """Real safe prompts from ToxicChat should be ALLOW."""
        safe = [s for s in toxicchat_samples if s.label == "safe"]
        assert len(safe) > 0, "No safe samples loaded"

        passed = 0
        failures: list[tuple[str, str, float]] = []

        for sample in safe:
            decision = await guard.classify(sample.prompt)
            print_decision(decision, show_prompt=True)
            if decision.action == Action.ALLOW:
                passed += 1
            else:
                failures.append((
                    sample.prompt[:80], decision.action.value, decision.ensemble_score,
                ))

        _print_test_summary("ToxicChat safe → ALLOW", len(safe), passed, failures)

        assert not failures, (
            f"{len(failures)}/{len(safe)} safe prompts were not ALLOW"
        )


# ---------------------------------------------------------------------------
# Tests: Mixed dataset
# ---------------------------------------------------------------------------

@skip_no_config
@skip_no_api_keys
@skip_no_datasets
@skip_no_hf_token
class TestMixedDataset:
    """Combined Qualifire + ToxicChat tests."""

    async def test_batch_separation(
        self, guard: InjectionGuard, mixed_samples: list[TestSample],
    ):
        """Attack prompts (injection + toxic) should score higher than benign on average."""
        prompts = [s.prompt for s in mixed_samples]
        decisions = await guard.classify_batch(prompts)
        print_batch(decisions, prompts=prompts)

        attack_scores = [
            d.ensemble_score
            for d, s in zip(decisions, mixed_samples)
            if s.is_attack
        ]
        benign_scores = [
            d.ensemble_score
            for d, s in zip(decisions, mixed_samples)
            if not s.is_attack
        ]

        avg_attack = sum(attack_scores) / len(attack_scores) if attack_scores else 0
        avg_benign = sum(benign_scores) / len(benign_scores) if benign_scores else 1

        from rich.console import Console
        from rich.panel import Panel

        sep = avg_attack - avg_benign
        sep_color = "green" if sep > 0 else "red"
        Console().print(Panel(
            f"[bold]Attack:[/bold] {len(attack_scores)} samples, avg score: {avg_attack:.3f}\n"
            f"[bold]Benign:[/bold] {len(benign_scores)} samples, avg score: {avg_benign:.3f}\n"
            f"[bold]Separation:[/bold] [{sep_color}]{sep:+.3f}[/{sep_color}]",
            title="Mixed Batch Score Separation",
            border_style="blue",
        ))

        assert avg_attack > avg_benign, (
            f"Attack avg ({avg_attack:.3f}) should exceed benign avg ({avg_benign:.3f})"
        )


# ---------------------------------------------------------------------------
# Tests: Stage 1 signal enrichment
# ---------------------------------------------------------------------------

@skip_no_config
@skip_no_api_keys
@skip_no_datasets
@skip_no_hf_token
class TestStage1Signals:
    """Verify Stage 1 classifiers enrich the SignalVector."""

    @pytest.mark.skipif(
        not (prereqs.has_safeguard or prereqs.has_litguard or prereqs.has_gcp_project),
        reason="No Stage 1 services reachable (need Safeguard, litguard, or Model Armor)",
    )
    async def test_stage1_signals_populated(
        self, guard: InjectionGuard, qualifire_samples: list[TestSample],
    ):
        """At least one Stage 1 signal should be populated."""
        injections = [s for s in qualifire_samples if s.label == "injection"]
        assert len(injections) > 0

        sample = injections[0]
        decision = await guard.classify(sample.prompt)
        print_decision(decision, show_prompt=True)

        signals = decision.preprocessor.signals
        s1 = signals.stage_one

        has_deberta = s1.deberta_label is not None
        has_model_armor = s1.model_armor_blocked is not None
        has_safeguard = s1.safeguard_violation is not None

        assert has_deberta or has_model_armor or has_safeguard, (
            "Expected at least one Stage 1 signal (DeBERTa, Model Armor, or Safeguard) "
            "to be populated in the SignalVector. "
            f"DeBERTa={has_deberta}, Model Armor={has_model_armor}, Safeguard={has_safeguard}"
        )

        if has_safeguard and s1.safeguard_violation:
            from rich.console import Console
            from rich.panel import Panel
            Console().print(Panel(
                f"[bold]Categories:[/bold] {s1.safeguard_categories}\n"
                f"[bold]Reasoning:[/bold] {s1.safeguard_reasoning}",
                title="Safeguard Signal",
                border_style="yellow",
            ))

    async def test_decision_audit_completeness(
        self, guard: InjectionGuard, qualifire_samples: list[TestSample],
    ):
        """Every Decision field should be populated."""
        sample = qualifire_samples[0]
        decision = await guard.classify(sample.prompt)
        print_decision(decision, show_prompt=True)

        assert decision.action in (Action.ALLOW, Action.FLAG, Action.BLOCK)
        assert 0.0 <= decision.ensemble_score <= 1.0
        assert decision.preprocessor is not None
        assert len(decision.router_path) >= 1
        assert decision.latency_ms > 0
        assert len(decision.model_scores) >= 1
