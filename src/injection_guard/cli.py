"""CLI entry point for injection-guard."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from CWD to find the project root (contains pyproject.toml)."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _run_tests(args: argparse.Namespace) -> int:
    """Run the test suite via pytest."""
    try:
        import pytest
    except ImportError:
        print("pytest is required. Install with: pip install pytest", file=sys.stderr)
        return 1

    root = _find_project_root()
    pytest_args: list[str] = []

    if args.suite == "unit":
        pytest_args.append(str(root / "tests" / "unit"))
    elif args.suite == "integration":
        pytest_args.append(str(root / "tests" / "integration"))
        if not args.benchmark:
            pytest_args.extend(["-m", "not benchmark"])
    elif args.suite == "benchmark":
        pytest_args.append(str(root / "tests" / "integration"))
        pytest_args.extend(["-m", "benchmark"])
    elif args.suite == "all":
        pytest_args.append(str(root / "tests"))
        if not args.benchmark:
            pytest_args.extend(["-m", "not benchmark"])

    if args.verbose:
        pytest_args.append("-v")

    if args.coverage:
        pytest_args.extend(["--cov=injection_guard", "--cov-report=term-missing"])

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    pytest_args.extend(args.extra)

    return pytest.main(pytest_args)


def _run_classify(args: argparse.Namespace) -> int:
    """Classify a single prompt."""
    from injection_guard.guard import InjectionGuard
    from injection_guard.reporting import print_decision

    config_path = args.config
    if config_path:
        guard = InjectionGuard.from_config(config_path)
    else:
        from injection_guard.classifiers.regex import RegexPrefilter
        from injection_guard.router.cascade import CascadeRouter
        from injection_guard.types import CascadeConfig
        guard = InjectionGuard(
            classifiers=[RegexPrefilter()],
            router=CascadeRouter(CascadeConfig()),
        )

    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read().strip()

    decision = asyncio.run(guard.classify(prompt))
    print_decision(decision, show_prompt=args.show_prompt)
    return 0 if decision.action.value == "allow" else 1


def _run_batch(args: argparse.Namespace) -> int:
    """Classify prompts from a file (one per line)."""
    from injection_guard.guard import InjectionGuard
    from injection_guard.reporting import print_batch

    config_path = args.config
    if config_path:
        guard = InjectionGuard.from_config(config_path)
    else:
        from injection_guard.classifiers.regex import RegexPrefilter
        from injection_guard.router.cascade import CascadeRouter
        from injection_guard.types import CascadeConfig
        guard = InjectionGuard(
            classifiers=[RegexPrefilter()],
            router=CascadeRouter(CascadeConfig()),
        )

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    prompts = [
        line.strip()
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]

    decisions = asyncio.run(guard.classify_batch(prompts))
    print_batch(decisions, prompts=prompts)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="injection-guard",
        description="Prompt injection detection toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- test ---
    test_parser = subparsers.add_parser("test", help="Run test suites")
    test_parser.add_argument(
        "suite",
        nargs="?",
        default="unit",
        choices=["unit", "integration", "benchmark", "all"],
        help="Which test suite to run (default: unit)",
    )
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    test_parser.add_argument("--coverage", action="store_true", help="Enable coverage report")
    test_parser.add_argument("--benchmark", action="store_true", help="Include benchmark tests")
    test_parser.add_argument("-k", "--keyword", help="pytest -k filter expression")
    test_parser.add_argument("extra", nargs="*", help="Extra args passed to pytest")

    # --- classify ---
    classify_parser = subparsers.add_parser("classify", help="Classify a single prompt")
    classify_parser.add_argument("prompt", help="Prompt text (use '-' for stdin)")
    classify_parser.add_argument("-c", "--config", help="Path to YAML config file")
    classify_parser.add_argument("--show-prompt", action="store_true", help="Include prompt in output")

    # --- batch ---
    batch_parser = subparsers.add_parser("batch", help="Classify prompts from a file")
    batch_parser.add_argument("file", help="File with one prompt per line")
    batch_parser.add_argument("-c", "--config", help="Path to YAML config file")

    args = parser.parse_args(argv)

    if args.command == "test":
        return _run_tests(args)
    elif args.command == "classify":
        return _run_classify(args)
    elif args.command == "batch":
        return _run_batch(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
