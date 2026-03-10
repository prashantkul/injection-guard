"""Rich-powered reporting for injection-guard."""
from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from injection_guard.types import Action, ClassifierResult, Decision, EvalMetrics

__all__ = [
    "print_decision",
    "print_batch",
    "print_benchmark",
    "print_confusion_matrix",
]

console = Console()

_ACTION_STYLES = {
    Action.ALLOW: "bold green",
    Action.FLAG: "bold yellow",
    Action.BLOCK: "bold red",
}

_ACTION_ICONS = {
    Action.ALLOW: "[green]ALLOW[/green]",
    Action.FLAG: "[yellow]FLAG[/yellow]",
    Action.BLOCK: "[red]BLOCK[/red]",
}


def _score_color(score: float) -> str:
    """Return a Rich color string based on score severity."""
    if score >= 0.85:
        return "red"
    if score >= 0.50:
        return "yellow"
    return "green"


def print_decision(decision: Decision, *, show_prompt: bool = False) -> None:
    """Pretty-print a single classification Decision.

    Args:
        decision: The Decision object to display.
        show_prompt: If True, include the normalized prompt text.
    """
    action_text = _ACTION_ICONS.get(decision.action, str(decision.action))
    score_color = _score_color(decision.ensemble_score)

    header = Text()
    header.append("Action: ")
    header.append(decision.action.value.upper(), style=_ACTION_STYLES.get(decision.action, ""))
    header.append(f"  Score: ")
    header.append(f"{decision.ensemble_score:.3f}", style=f"bold {score_color}")
    header.append(f"  Latency: {decision.latency_ms:.1f}ms")
    if decision.degraded:
        header.append("  [DEGRADED]", style="bold red")

    console.print(header)

    if decision.model_scores:
        table = Table(title="Model Scores", show_header=True, header_style="bold cyan")
        table.add_column("Classifier", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Label")
        table.add_column("Confidence", justify="right")

        for name, result in decision.model_scores.items():
            sc = _score_color(result.score)
            label_style = "red" if result.label == "injection" else "green"
            error = result.metadata.get("error")
            name_display = f"{name} [red](err)[/red]" if error else name

            table.add_row(
                name_display,
                f"[{sc}]{result.score:.3f}[/{sc}]",
                f"[{label_style}]{result.label}[/{label_style}]",
                f"{result.confidence:.2f}",
            )
        console.print(table)

    if decision.router_path:
        console.print(f"[dim]Router path:[/dim] {' → '.join(decision.router_path)}")

    if decision.reasoning:
        console.print(f"[dim]Reasoning:[/dim] {decision.reasoning}")

    if show_prompt and decision.preprocessor.normalized_prompt:
        console.print(
            Panel(
                decision.preprocessor.normalized_prompt[:500],
                title="Normalized Prompt",
                border_style="dim",
            )
        )

    console.print()


def print_batch(decisions: Sequence[Decision], *, prompts: Sequence[str] | None = None) -> None:
    """Print a summary table for batch classification results.

    Args:
        decisions: List of Decision objects from classify_batch.
        prompts: Optional list of original prompts for display.
    """
    table = Table(title="Batch Classification Results", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", justify="right")
    if prompts:
        table.add_column("Prompt", max_width=50, no_wrap=True)
    table.add_column("Action", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Latency", justify="right", style="dim")
    table.add_column("Path", style="dim")

    for i, decision in enumerate(decisions):
        action_style = _ACTION_STYLES.get(decision.action, "")
        sc = _score_color(decision.ensemble_score)

        row = [str(i + 1)]
        if prompts:
            prompt_text = prompts[i][:47] + "..." if len(prompts[i]) > 50 else prompts[i]
            row.append(prompt_text)
        row.extend([
            f"[{action_style}]{decision.action.value.upper()}[/{action_style}]",
            f"[{sc}]{decision.ensemble_score:.3f}[/{sc}]",
            f"{decision.latency_ms:.0f}ms",
            " → ".join(decision.router_path),
        ])
        table.add_row(*row)

    console.print(table)

    # Summary
    from collections import Counter
    counts = Counter(d.action for d in decisions)
    avg_latency = sum(d.latency_ms for d in decisions) / len(decisions) if decisions else 0

    summary = Text()
    summary.append(f"\nTotal: {len(decisions)}  ")
    for action in (Action.ALLOW, Action.FLAG, Action.BLOCK):
        count = counts.get(action, 0)
        style = _ACTION_STYLES.get(action, "")
        summary.append(f"{action.value.upper()}: ", style=style)
        summary.append(f"{count}  ")
    summary.append(f"Avg latency: {avg_latency:.1f}ms", style="dim")

    console.print(summary)
    console.print()


def print_confusion_matrix(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """Print a rich confusion matrix.

    Args:
        tp: True positives (correctly detected injections).
        fp: False positives (benign marked as injection).
        tn: True negatives (correctly allowed benign).
        fn: False negatives (missed injections).
        title: Table title.
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("", style="bold")
    table.add_column("Pred: Injection", justify="center")
    table.add_column("Pred: Benign", justify="center")

    table.add_row(
        "Actual: Injection",
        f"[bold green]{tp}[/bold green] (TP)",
        f"[bold red]{fn}[/bold red] (FN)",
    )
    table.add_row(
        "Actual: Benign",
        f"[bold red]{fp}[/bold red] (FP)",
        f"[bold green]{tn}[/bold green] (TN)",
    )

    console.print(table)
    console.print()


def print_benchmark(
    decisions: Sequence[Decision],
    labels: Sequence[str],
    *,
    title: str = "Benchmark Results",
) -> None:
    """Print full benchmark report with confusion matrix and metrics.

    Args:
        decisions: List of Decision objects.
        labels: Ground-truth labels ("injection"/"jailbreak" or "benign"),
                one per decision.
        title: Report title.
    """
    injection_labels = {"injection", "jailbreak"}

    tp = fp = tn = fn = 0
    injection_scores: list[float] = []
    benign_scores: list[float] = []

    for decision, label in zip(decisions, labels):
        predicted_injection = decision.action in (Action.FLAG, Action.BLOCK)
        actual_injection = label.lower() in injection_labels

        if predicted_injection and actual_injection:
            tp += 1
        elif predicted_injection and not actual_injection:
            fp += 1
        elif not predicted_injection and actual_injection:
            fn += 1
        else:
            tn += 1

        if actual_injection:
            injection_scores.append(decision.ensemble_score)
        else:
            benign_scores.append(decision.ensemble_score)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    avg_inj = sum(injection_scores) / len(injection_scores) if injection_scores else 0.0
    avg_ben = sum(benign_scores) / len(benign_scores) if benign_scores else 0.0

    console.rule(f"[bold]{title}[/bold]")
    console.print()

    # Confusion matrix
    print_confusion_matrix(tp, fp, tn, fn)

    # Metrics table
    metrics = Table(title="Classification Metrics", show_header=True, header_style="bold cyan")
    metrics.add_column("Metric", style="bold")
    metrics.add_column("Value", justify="right")

    def _metric_color(val: float, good_above: float = 0.7) -> str:
        return "green" if val >= good_above else ("yellow" if val >= 0.4 else "red")

    metrics.add_row("Accuracy", f"[{_metric_color(accuracy)}]{accuracy:.3f}[/{_metric_color(accuracy)}]")
    metrics.add_row("Precision", f"[{_metric_color(precision)}]{precision:.3f}[/{_metric_color(precision)}]")
    metrics.add_row("Recall", f"[{_metric_color(recall)}]{recall:.3f}[/{_metric_color(recall)}]")
    metrics.add_row("F1 Score", f"[{_metric_color(f1)}]{f1:.3f}[/{_metric_color(f1)}]")
    metrics.add_row("False Positive Rate", f"[{_metric_color(1 - fpr)}]{fpr:.3f}[/{_metric_color(1 - fpr)}]")
    metrics.add_row("False Negative Rate", f"[{_metric_color(1 - fnr)}]{fnr:.3f}[/{_metric_color(1 - fnr)}]")

    console.print(metrics)
    console.print()

    # Score distribution
    dist = Table(title="Score Distribution", show_header=True, header_style="bold cyan")
    dist.add_column("Class", style="bold")
    dist.add_column("Count", justify="right")
    dist.add_column("Avg Score", justify="right")

    dist.add_row(
        "[red]Injection[/red]",
        str(len(injection_scores)),
        f"[{_score_color(avg_inj)}]{avg_inj:.3f}[/{_score_color(avg_inj)}]",
    )
    dist.add_row(
        "[green]Benign[/green]",
        str(len(benign_scores)),
        f"[{_score_color(avg_ben)}]{avg_ben:.3f}[/{_score_color(avg_ben)}]",
    )

    console.print(dist)

    # Action breakdown
    from collections import Counter
    action_counts = Counter(d.action for d in decisions)
    avg_latency = sum(d.latency_ms for d in decisions) / len(decisions) if decisions else 0

    console.print()
    breakdown = Text()
    breakdown.append("Actions: ")
    for action in (Action.ALLOW, Action.FLAG, Action.BLOCK):
        count = action_counts.get(action, 0)
        style = _ACTION_STYLES.get(action, "")
        breakdown.append(f"{action.value.upper()}={count} ", style=style)
    breakdown.append(f" | Avg latency: {avg_latency:.1f}ms", style="dim")
    console.print(breakdown)

    console.rule()
    console.print()
