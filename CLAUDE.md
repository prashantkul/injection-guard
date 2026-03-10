# injection-guard — Claude Code Instructions

## Project Overview
Prompt injection detection library with ensemble classifier architecture.
Pure Python, async-first, pluggable classifiers (API + local models).

## Key Conventions
- Python 3.10+, async-first with sync wrappers
- All types in `src/injection_guard/types.py` — single source of truth
- No circular imports: types.py has ZERO internal imports
- Use `from __future__ import annotations` in every file
- Pydantic v2 for config validation, dataclasses for internal data
- All async functions use `async def`, provide `_sync` wrappers via `asyncio.run()`

## Architecture Layers (dependency order)
1. types.py — depends on nothing internal
2. preprocessor/ — depends on types.py only (gliner.py requires gliner package)
3. gate/ — depends on types.py only (requires google-cloud-modelarmor)
4. classifiers/ — depends on types.py only
5. aggregator/ — depends on types.py only
6. router/ — depends on types.py, calls classifiers
7. engine.py — depends on types.py only
8. guard.py — orchestrates all above
9. eval/ — depends on all above

## Implementation Rules
- Every public function has a Google-style docstring
- Every module has `__all__` defined
- Use `typing.Protocol` for BaseClassifier, not ABC
- Classifiers receive `SignalVector | None` — never required
- Router returns `list[tuple[str, ClassifierResult]]` — ordered by invocation
- All API calls use `asyncio` with proper timeout handling
- Retries use exponential backoff: `delay = backoff_base_ms * (2 ** attempt)`
- JSON parsing from API models: strip markdown fences, validate schema, check label-score consistency

## Testing Rules
- Use pytest + pytest-asyncio
- Mock all external API calls (use `respx` for httpx, `unittest.mock` for others)
- Mock ONNX runtime with fake model that returns predetermined scores
- Every test file is independently runnable
- Preprocessor tests use known attack payloads as fixtures
- Target: 90%+ coverage on preprocessor and aggregator modules

## Error Handling
- Never raise on classifier failure — return ClassifierResult with score=0.5, confidence=0.0, metadata={"error": str}
- Router handles classifier exceptions and continues to next
- Decision.degraded=True when any classifier fails

## Build & Run
- `pip install -e ".[dev]"` for development
- `pytest tests/ -v` to run all tests
- `pytest tests/test_preprocessor/ -v` to run preprocessor tests only
