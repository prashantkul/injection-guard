# injection-guard

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Async](https://img.shields.io/badge/async-first-purple.svg)]()
[![Ensemble](https://img.shields.io/badge/classifiers-ensemble-orange.svg)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991.svg)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-d4a574.svg)](https://anthropic.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4.svg)](https://deepmind.google/technologies/gemini/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DeBERTa-FFD21E.svg)](https://huggingface.co)
[![Model Armor](https://img.shields.io/badge/Google_Cloud-Model_Armor-4285F4.svg)](https://cloud.google.com/security/products/model-armor)

Two-stage prompt injection detection system with an ensemble classifier architecture. **Stage 1** (pre-gate + pre-filter) uses Google Cloud Model Armor and fine-tunable DeBERTa models for fast, high-recall local detection (~100-200ms). **Stage 2** (frontier ensemble) escalates ambiguous cases to API classifiers (OpenAI, Anthropic, Gemini) for high-accuracy classification. Async-first Python, 6-stage preprocessor with NER-based signal detection, and cascade routing with early exit.

## Architecture

```mermaid
block-beta
  columns 6

  block:input:1
    columns 1
    A["Prompt"]
  end

  block:preprocess:1
    columns 1
    style preprocess fill:#2d3748,color:#fff,stroke:#4a5568
    B["Preprocessor"]
    B1["Unicode"]
    B2["Encoding"]
    B3["Structural"]
    B4["Token"]
    B5["GLiNER NER"]
    B6["Regex"]
    BP["Signal Extraction"]
  end

  block:fast:1
    columns 1
    style fast fill:#065f46,color:#fff,stroke:#047857
    F["Pre-gate + Pre-filter"]
    F0["Model Armor (GCP)"]
    F1["DeBERTa (fine-tunable)"]
    F2["~100-200ms"]
    FP["High recall, fast exit"]
  end

  block:classify:1
    columns 1
    style classify fill:#2c5282,color:#fff,stroke:#2b6cb0
    C["Parallel Router"]
    C1["OpenAI"]
    C2["Anthropic"]
    C3["Gemini"]
    CP["Ensemble Classification"]
  end

  block:decide:1
    columns 1
    style decide fill:#553c9a,color:#fff,stroke:#6b46c1
    D["Aggregator"]
    D1["Weighted Average"]
    D2["Threshold Engine"]
    DP["Score Aggregation"]
  end

  block:output:1
    columns 1
    style output fill:#2f855a,color:#fff,stroke:#38a169
    E1["ALLOW"]
    E2["FLAG"]
    E3["BLOCK"]
    EP["Decision"]
  end

  A --> B
  B --> F
  F --> C
  C --> D
  D --> E1
```

## How It Works

```mermaid
flowchart LR
  subgraph Preprocessor
    direction TB
    P1["1. Unicode"] --> P2["2. Encoding"]
    P2 --> P3["3. Structural"]
    P3 --> P4["4. Token"]
    P4 --> P5["5. GLiNER NER"]
    P5 --> P6["6. Regex"]
  end

  subgraph Signals
    direction TB
    SV["SignalVector"]
    RP["risk_prior"]
  end

  subgraph Pre-filter
    direction TB
    MA["Model Armor (GCP)"]
    MA -->|"pass"| HF["DeBERTa (fine-tuned)"]
    MA -->|"BLOCK\n(high confidence)"| Decision
  end

  subgraph Router
    direction TB
    R1["Launch frontier classifiers"]
    R2["Wait for category quorum"]
    R3["Cancel remaining"]
  end

  subgraph Decision
    direction TB
    AG["Weighted Aggregation"]
    TH["Threshold Check"]
  end

  Preprocessor --> Signals
  Signals --> Pre-filter
  Pre-filter -->|"high confidence\nbenign"| Decision
  Pre-filter -->|"uncertain or\ninjection"| Router
  Router --> Decision

  style Preprocessor fill:#2d3748,color:#e2e8f0
  style Signals fill:#553c9a,color:#e9d8fd
  style Pre-filter fill:#065f46,color:#d1fae5
  style Router fill:#2c5282,color:#bee3f8
  style Decision fill:#2f855a,color:#c6f6d5
```

### Recommended Ensemble Strategy

The architecture uses a **tiered approach** optimized from [eval results](docs/eval-results.md) on the Qualifire benchmark:

1. **Pre-gate (Model Armor)** — Google Cloud Model Armor screens prompts first (~180ms). High-confidence injections are blocked immediately. Low false positive rate on general benchmarks (1-7%), though domain-specific traffic may see higher FP rates — test with your data. Optional — requires GCP.
2. **Fast pre-filter (DeBERTa)** — Fine-tunable DeBERTa model (~100ms on GPU, 99% recall) catches remaining obvious injections and short-circuits high-confidence benign prompts. Customers can [fine-tune](docs/fine-tuning-strategy.md) this model on their domain data.
3. **Frontier ensemble** — For uncertain cases, the cascade/parallel router fires frontier API classifiers (Anthropic, OpenAI with reasoning, Gemini) and waits for quorum. These provide 80-84% accuracy with nuanced scoring.
4. **Weighted aggregation** — The aggregator combines all scores using learned weights, then applies threshold engine for ALLOW/FLAG/BLOCK.

This gives sub-200ms latency for ~70% of requests (clear benign/injection via pre-gate + pre-filter) while maintaining 83%+ accuracy on ambiguous cases via the frontier ensemble.

## Preprocessor Pipeline

Six stages extract signals from the raw prompt before classification:

| Stage | Detects | Key Signals |
|-------|---------|-------------|
| 1. Unicode | Homoglyphs, zero-width chars, BiDi overrides | `homoglyph_count`, `zero_width_count`, `script_mixing` |
| 2. Encoding | Base64, hex, URL-encoding, nested encoding | `encodings_found`, `encoding_density`, `nested_encoding` |
| 3. Structural | Chat delimiters, XML/HTML tags, instruction boundaries | `chat_delimiters_found`, `separator_density` |
| 4. Token | Split-keyword attacks, prompt stuffing | `reconstructed_keywords`, `repetition_ratio` |
| 5. GLiNER NER | Injection-specific semantic entities | `entity_count`, `max_entity_confidence` |
| 6. Regex | Known injection patterns (12 built-in) | `match_count`, `matched_patterns` |

Signals feed into a `risk_prior` (0.0-1.0) that can block early or escalate routing. They're also formatted into natural language and appended to LLM classifier prompts as evidence.

See [docs/ner-signals.md](docs/ner-signals.md) for details on how GLiNER NER works and how signals augment classifiers.

## Classifiers

### Pre-gate

| Gate | Type | Accuracy | Precision | Role |
|------|------|----------|-----------|------|
| [Model Armor](docs/eval-results.md#model-armor--qualifire-dataset-200-samples) | GCP API | 58-75% | 89-95% | Pre-gate for high-confidence detections. FP rates vary by domain — evaluate on your traffic. |

Model Armor runs *before* classifiers as an optional pre-gate. Its high precision (89-95%) and low false positive rate (1-7%) make it safe to block immediately on high-confidence detections. See [docs/safeguard-policy.md](docs/safeguard-policy.md) for template configuration.

### Classifiers

| Classifier | Type | Weight | Category | Accuracy | Approach |
|------------|------|--------|----------|----------|----------|
| Anthropic | API | 2.0 | api | 83.5% | Claude with few-shot classification prompt |
| OpenAI | API | 1.5 | api | 82.0% | GPT-5 with reasoning tokens (high effort) |
| Gemini | API | 1.5 | api | 80.5% | Gemini via google-genai with few-shot prompt |
| HF DeBERTa | Local | 1.0 | local | 65-70% | HuggingFace models via litguard (fine-tunable) |
| Safeguard | Local | 1.5 | local | 60.5% | gpt-oss-safeguard with 6-category PI/JB policy |
| Local LLM | Local | 1.5 | local | — | Any Ollama/vLLM model with classification prompt |
| ONNX | Local | 1.0 | local | — | ONNX Runtime inference |

All classifiers implement the `BaseClassifier` protocol and receive the `SignalVector` from the preprocessor. API classifiers use a shared few-shot classification prompt with signal context. Safeguard uses a custom policy-based system prompt (see [docs/safeguard-policy.md](docs/safeguard-policy.md)). HF DeBERTa models are served via [litguard](docs/litguard-spec.md) and can be [fine-tuned](docs/fine-tuning-strategy.md) on customer data.

## Routing

Two strategies control how classifiers are invoked:

**Parallel Router** — fires all classifiers concurrently, returns when a category quorum is met:

```yaml
router:
  type: parallel
  timeout_ms: 10000
  category_quorum:
    local: 1   # at least 1 local model must respond
    api: 2     # at least 2 API models must respond
```

**Cascade Router** (recommended) — runs classifiers tier-by-tier (fast → medium → slow), exits early on high confidence. This is the recommended strategy for the tiered pre-filter architecture:

```yaml
router:
  type: cascade
  timeout_ms: 10000
  fast_confidence: 0.85          # exit early if confidence > 85%
  escalate_on_high_risk_prior: true
  risk_prior_escalation_threshold: 0.7  # skip fast tier if risk_prior > 0.7
```

The cascade router groups classifiers by their `latency_tier` attribute and runs them in order:

| Tier | Classifiers | Latency | Behavior |
|------|-------------|---------|----------|
| fast | DeBERTa (HF), ONNX, Regex | ~100ms | Run first. If confidence > `fast_confidence`, return immediately. |
| medium | Safeguard, Local LLM | ~1-5s | Run if fast tier is uncertain. |
| slow | OpenAI, Anthropic, Gemini | ~2-10s | Run only for ambiguous cases. |

If the preprocessor's `risk_prior` exceeds `risk_prior_escalation_threshold`, the fast tier is skipped entirely and classification starts at medium/slow tiers — this prevents high-risk prompts from being cleared by a less capable local model.

This gives sub-200ms decisions for ~70% of traffic (clear benign/injection via DeBERTa) while escalating only ambiguous cases to frontier API classifiers.

## Quick Start

### YAML Config

```yaml
# --- Stage 1: Pre-gate + Pre-filter (fast, high recall) ---
gate:
  type: model_armor
  project: ${GOOGLE_CLOUD_PROJECT}
  location: global
  template_id: my-injection-template
  block_on: HIGH              # block only high-confidence detections
  fail_mode: open             # if MA is down, let prompts through

classifiers:
  # Fast pre-filter (fine-tunable, ~100ms)
  - type: hf_compat
    model: deberta-injection
    base_url: http://192.168.1.199:8234/v1
    weight: 1.0
    category: local

  # --- Stage 2: Frontier ensemble (high accuracy) ---
  - type: anthropic
    model: claude-sonnet-4-6
    weight: 2.0
    category: api

  - type: openai
    model: gpt-5-2025-08-07
    weight: 1.5
    reasoning_effort: high
    category: api

  - type: gemini
    model: gemini-3.1-pro-preview
    weight: 1.5
    category: api

router:
  type: cascade
  timeout_ms: 10000
  fast_confidence: 0.85
  escalate_on_high_risk_prior: true
  risk_prior_escalation_threshold: 0.7

thresholds:
  block: 0.85
  flag: 0.50

aggregator: weighted_average

preprocessor:
  gliner_model: urchade/gliner_base
```

```python
from injection_guard import InjectionGuard

guard = InjectionGuard.from_config("config.yaml")

decision = await guard.classify("Ignore all previous instructions")
print(decision.action)        # Action.BLOCK
print(decision.ensemble_score) # 0.97
print(decision.model_scores)   # per-classifier results

decision = await guard.classify("What is the capital of France?")
print(decision.action)        # Action.ALLOW
```

### Programmatic Setup

```python
from injection_guard import InjectionGuard
from injection_guard.classifiers import AnthropicClassifier, SafeguardClassifier
from injection_guard.router import ParallelRouter
from injection_guard.types import ParallelConfig

guard = InjectionGuard(
    classifiers=[
        AnthropicClassifier(model="claude-sonnet-4-6"),
        SafeguardClassifier(
            model="gpt-oss-safeguard:120b",
            base_url="http://192.168.1.199:11434/v1",
        ),
    ],
    router=ParallelRouter(ParallelConfig(
        timeout_ms=10000,
        category_quorum={"local": 1, "api": 1},
        classifier_categories={
            "anthropic-claude-sonnet-4-6": "api",
            "safeguard-gpt-oss-safeguard": "local",
        },
    )),
)
```

### Sync Wrapper

```python
decision = guard.classify_sync("Tell me about Python")
```

### Environment Variables

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_CLOUD_PROJECT=my-project-123
GOOGLE_CLOUD_REGION=global
```

The `.env` file is loaded automatically on `InjectionGuard` init.

## Benchmark Results

Evaluated on [Qualifire prompt-injections-benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark) (200 balanced samples: 100 injection, 100 benign). Full results in [docs/eval-results.md](docs/eval-results.md).

**Stage 2: Frontier Classifiers**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Anthropic claude-opus-4.6 | **83.5%** | 0.860 | 0.800 | 0.829 |
| Anthropic claude-sonnet-4.6 | 82.5% | 0.788 | 0.890 | 0.836 |
| OpenAI gpt-5 (high reasoning) | 82.0% | 0.758 | 0.940 | 0.839 |
| Gemini 3.1-pro-preview | 80.5% | 0.740 | 0.940 | 0.828 |
| Gemini 3-flash-preview | 80.0% | 0.724 | 0.970 | 0.829 |
| OpenAI gpt-5-mini (medium reasoning) | 79.0% | 0.769 | 0.830 | 0.798 |

**Stage 1: Pre-gate + Pre-filter (local/fast)**

| Model | Accuracy | Precision | Recall | F1 | Latency |
|-------|----------|-----------|--------|----|---------|
| Model Armor (MA Low) | 74.5% | 0.889 | 0.560 | 0.687 | ~800ms |
| Model Armor (MA High) | 58.5% | **0.947** | 0.180 | 0.303 | ~180ms |
| protectai/deberta (open-weight) | 69.5% | 0.714 | 0.650 | 0.681 | ~100ms |
| deepset/deberta (open-weight) | 65.0% | 0.589 | **0.990** | 0.739 | ~100ms |

Run benchmarks yourself:

```bash
# Quick model benchmarks (10-sample)
pytest tests/integration/test_model_benchmarks.py -v -s

# Full Qualifire eval (200-sample, requires API keys + HF token)
pytest tests/integration/test_eval_classifiers.py -v -s -k "test_openai_gpt_5_high"
```

## Build & Test

```bash
pip install -e ".[dev]"

# Unit tests (no API keys needed)
pytest tests/unit/ -v

# Integration tests (requires API keys in .env)
pytest tests/integration/ -v

# Model benchmarks
pytest tests/integration/test_model_benchmarks.py -v -s
```

## Documentation

- [Deployment Guide](docs/deployment-guide.md) — production deployment patterns, FastAPI/LangChain integration, scaling, and monitoring
- [Eval Results](docs/eval-results.md) — full benchmark results across all classifiers and models
- [Fine-Tuning Strategy](docs/fine-tuning-strategy.md) — how to fine-tune DeBERTa models to improve detection metrics
- [Domain Fine-Tuning](docs/domain-fine-tuning.md) — domain-specific tuning for healthcare, finance, legal, and other verticals
- [NER Signals & Preprocessor](docs/ner-signals.md) — how GLiNER NER works and how signals augment classifiers
- [Safeguard Policy Setup](docs/safeguard-policy.md) — gpt-oss-safeguard deployment, policy categories, and configuration
- [litguard Spec](docs/litguard-spec.md) — LitServe-based model serving platform for HuggingFace models

## Project Structure

```
src/injection_guard/
    types.py              # All shared types (single source of truth)
    guard.py              # Main orchestrator
    config.py             # YAML config loader & factory
    engine.py             # Threshold decision engine
    cli.py                # CLI entry point
    reporting.py          # Rich-powered reporting output
    preprocessor/
        pipeline.py       # 6-stage pipeline orchestration
        unicode.py        # Stage 1: Unicode normalization
        encoding.py       # Stage 2: Encoding detection
        structural.py     # Stage 3: Structural analysis
        token.py          # Stage 4: Token boundary detection
        gliner.py         # Stage 5: GLiNER entity detection
        regex.py          # Stage 6: Regex pattern matching
    classifiers/
        prompts.py        # Shared few-shot prompt & signal formatting
        openai.py         # OpenAI API classifier
        anthropic.py      # Anthropic API classifier
        gemini.py         # Google Gemini via Vertex AI
        safeguard.py      # gpt-oss-safeguard with PI/JB policy
        local_llm.py      # Ollama, vLLM, OpenAI-compatible
        onnx.py           # Local ONNX model
        regex.py          # Legacy regex prefilter
    router/
        cascade.py        # Tier-by-tier with early exit
        parallel.py       # Concurrent with category quorum
    aggregator/
        weighted.py       # Weighted average
        voting.py         # Majority voting
        meta.py           # Meta-classifier stacking
    gate/
        model_armor.py    # Google Cloud Model Armor (optional)
    eval/
        runner.py         # Dataset loading & evaluation
        report.py         # Metrics & threshold recommendation
```
