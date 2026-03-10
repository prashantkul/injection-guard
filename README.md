# injection-guard

Prompt injection detection library with an ensemble classifier architecture.
Async-first Python, pluggable classifiers (regex, ONNX, OpenAI, Anthropic), and a full evaluation toolkit.

## Architecture Overview

```mermaid
graph TB
    subgraph Foundation
        T[types.py]
    end

    subgraph Analysis
        PP[Preprocessor Pipeline]
        GA[Model Armor Gate]
    end

    subgraph Classification
        CL[Classifiers]
        RO[Router]
    end

    subgraph Decision
        AG[Aggregator]
        EN[Threshold Engine]
    end

    subgraph Orchestration
        GU[InjectionGuard]
    end

    subgraph Evaluation
        EV[Eval Runner & Report]
    end

    T --> PP & GA & CL & AG & EN
    PP --> GU
    GA --> GU
    CL --> RO --> GU
    AG --> GU
    EN --> GU
    GU --> EV
```

## Request Lifecycle

Every call to `InjectionGuard.classify()` follows this pipeline:

```mermaid
flowchart TD
    A[User Prompt] --> B[Preprocessor Pipeline]

    B --> B1[1. Unicode Normalization]
    B1 --> B2[2. Encoding Detection]
    B2 --> B3[3. Structural Analysis]
    B3 --> B4[4. Token Boundary Detection]
    B4 --> B5[5. GLiNER Entity Detection]
    B5 --> B6[Compute Risk Prior]

    B6 --> C{risk_prior >= \npreprocessor_block_threshold?}
    C -- Yes --> BLOCK1[BLOCK early exit]
    C -- No --> D{Model Armor\nenabled?}

    D -- Yes --> E[Model Armor Gate]
    E --> F{HIGH confidence\nPI + Jailbreak?}
    F -- Yes --> BLOCK2[BLOCK early exit]
    F -- No / MEDIUM --> G[Boost risk_prior]
    G --> H[Router]
    D -- No --> H

    H --> I[Classifiers]
    I --> J[Aggregator]
    J --> K[Threshold Engine]

    K --> L{ensemble_score}
    L -- ">= block_threshold" --> BLOCK3[BLOCK]
    L -- ">= flag_threshold" --> FLAG[FLAG]
    L -- "< flag_threshold" --> ALLOW[ALLOW]

    BLOCK3 --> Z[Decision]
    FLAG --> Z
    ALLOW --> Z
    BLOCK1 --> Z
    BLOCK2 --> Z
```

## Preprocessor Pipeline

Five stages extract signal vectors from the raw prompt before classification.

```mermaid
flowchart LR
    Raw[Raw Prompt] --> S1

    subgraph Pipeline
        S1[Unicode\nNormalizer] --> S2[Encoding\nDetector]
        S2 --> S3[Structural\nAnalyzer]
        S3 --> S4[Token Boundary\nDetector]
        S4 --> S5[GLiNER\nEntity Detector]
    end

    S1 -. UnicodeSignals .-> SV
    S2 -. EncodingSignals .-> SV
    S3 -. StructuralSignals .-> SV
    S4 -. TokenSignals .-> SV
    S5 -. LinguisticSignals .-> SV

    SV[SignalVector] --> RP[Risk Prior\nComputation]
    RP --> OUT[PreprocessorOutput]
```

| Stage | Detects | Key Signals |
|-------|---------|-------------|
| Unicode | Homoglyphs, zero-width chars, bidi overrides, script mixing | `homoglyph_count`, `zero_width_count`, `script_mixing` |
| Encoding | Base64, hex, URL-encoding, HTML entities, nesting | `encodings_found`, `encoding_density`, `nested_encoding` |
| Structural | Chat delimiters, XML/HTML tags, instruction boundaries | `chat_delimiters_found`, `separator_density` |
| Token | Split-keyword attacks, prompt stuffing, repetition | `reconstructed_keywords`, `repetition_ratio` |
| GLiNER | Injection-specific named entities (instruction override, role assignment, etc.) | `entity_count`, `max_entity_confidence` |

## Routing Strategies

Two router implementations control how classifiers are invoked.

```mermaid
flowchart TD
    subgraph CascadeRouter
        direction TB
        CR1[Group by latency tier] --> CR2{High risk_prior?}
        CR2 -- Yes --> CR3[Skip fast tier]
        CR2 -- No --> CR4[Run fast tier]
        CR3 --> CR5[Run medium tier]
        CR4 --> CRC{Confident?}
        CRC -- Yes --> CRX[Return early]
        CRC -- No --> CR5
        CR5 --> CRC2{Confident?}
        CRC2 -- Yes --> CRX
        CRC2 -- No --> CR6[Run slow tier]
        CR6 --> CRX
    end

    subgraph ParallelRouter
        direction TB
        PR1[Launch all classifiers\nconcurrently] --> PR2{Quorum\nagreement?}
        PR2 -- Yes --> PR3[Cancel remaining\ntasks]
        PR2 -- Timeout --> PR3
        PR3 --> PR4[Return results]
    end
```

## Classifiers

All classifiers implement the `BaseClassifier` protocol.

```mermaid
classDiagram
    class BaseClassifier {
        <<Protocol>>
        +name: str
        +latency_tier: fast | medium | slow
        +weight: float
        +classify(prompt, signals) ClassifierResult
    }

    class RegexPrefilter {
        +tier: fast
        +weight: 0.5
        +patterns: list
    }

    class OnnxClassifier {
        +tier: fast
        +weight: 1.0
        +model_path: str
    }

    class OpenAIClassifier {
        +tier: medium
        +weight: 1.5
        +model: str
    }

    class AnthropicClassifier {
        +tier: slow
        +weight: 2.0
        +model: str
    }

    BaseClassifier <|.. RegexPrefilter
    BaseClassifier <|.. OnnxClassifier
    BaseClassifier <|.. OpenAIClassifier
    BaseClassifier <|.. AnthropicClassifier
```

| Classifier | Tier | Weight | Approach |
|------------|------|--------|----------|
| RegexPrefilter | fast | 0.5 | Keyword pattern matching |
| OnnxClassifier | fast | 1.0 | Local ONNX Runtime inference |
| OpenAIClassifier | medium | 1.5 | OpenAI chat completion API |
| AnthropicClassifier | slow | 2.0 | Anthropic messages API |

## Aggregation Strategies

```mermaid
flowchart LR
    R["Classifier Results\n(name, score, weight)"] --> A{Aggregator\nStrategy}

    A -- Weighted Average --> WA["sum(score * weight)\n/ sum(weight)"]
    A -- Majority Voting --> MV["injection_votes\n/ total_votes"]
    A -- Meta Classifier --> MC["Learned stacking\nmodel prediction"]

    WA --> ES[ensemble_score]
    MV --> ES
    MC --> ES
```

## Decision Engine

```mermaid
flowchart LR
    ES[ensemble_score] --> T{Thresholds}
    T -- ">= 0.85 (block)" --> BLOCK[BLOCK]
    T -- ">= 0.50 (flag)" --> FLAG[FLAG]
    T -- "< 0.50" --> ALLOW[ALLOW]
```

Thresholds are configurable at init and updatable at runtime via `update_thresholds()`.

## Model Armor Gate (Optional)

Pre-screens prompts via Google Cloud Model Armor before ensemble classification.

```mermaid
flowchart TD
    P[Prompt] --> MA[Model Armor API]
    MA --> R{Response}
    R -- "PI+Jailbreak: HIGH" --> BLOCK[BLOCK early exit]
    R -- "PI+Jailbreak: MEDIUM" --> BOOST[Boost risk_prior +0.3]
    R -- "No match" --> PASS[Continue to router]
    R -- "Error + fail_mode=closed" --> BLOCK
    R -- "Error + fail_mode=open" --> PASS
```

## Evaluation Toolkit

```mermaid
flowchart LR
    DS[Dataset\nJSONL / CSV] --> ER[EvalRunner]
    ER --> G[InjectionGuard]
    G --> ER
    ER --> REP[EvalReport]

    REP --> M1[Confusion Matrix]
    REP --> M2[Precision / Recall / F1]
    REP --> M3[ROC-AUC]
    REP --> M4[Score Distribution]
    REP --> M5[Calibration Curves]
    REP --> M6[Per-Model Diagnostics]
    REP --> M7[Threshold Recommendation]
```

## Error Handling

```mermaid
flowchart TD
    CF[Classifier Call] --> E{Exception?}
    E -- No --> R[Return ClassifierResult]
    E -- Yes --> D[Return ClassifierResult\nscore=0.5, confidence=0.0\nmetadata.error = str]

    D --> RO[Router continues\nto next classifier]
    RO --> AG[Aggregator includes\ndegraded result]
    AG --> DEC["Decision.degraded = True"]
```

Failures never propagate -- every classifier error is captured and the pipeline continues.

## Quick Start

```python
from injection_guard import InjectionGuard
from injection_guard.classifiers import RegexPrefilter, AnthropicClassifier
from injection_guard.router import CascadeRouter

guard = InjectionGuard(
    classifiers=[RegexPrefilter(), AnthropicClassifier()],
    router=CascadeRouter(),
)

# Async
decision = await guard.classify("Ignore all previous instructions")
print(decision.action)   # Action.BLOCK
print(decision.reasoning)

# Sync
decision = guard.classify_sync("Hello, how are you?")
print(decision.action)   # Action.ALLOW
```

## Build & Test

```bash
pip install -e ".[dev]"    # Install with dev dependencies
pytest tests/ -v           # Run all 225 tests
```

## Project Structure

```
src/injection_guard/
    types.py              # All shared types (single source of truth)
    guard.py              # Main orchestrator
    engine.py             # Threshold decision engine
    preprocessor/
        pipeline.py       # 5-stage pipeline orchestration
        unicode.py        # Stage 1: Unicode normalization
        encoding.py       # Stage 2: Encoding detection
        structural.py     # Stage 3: Structural analysis
        token.py          # Stage 4: Token boundary detection
        gliner.py         # Stage 5: GLiNER entity detection
    classifiers/
        regex.py          # Fast regex prefilter
        onnx.py           # Local ONNX model
        openai.py         # OpenAI API classifier
        anthropic.py      # Anthropic API classifier
    router/
        cascade.py        # Tier-by-tier with early exit
        parallel.py       # Concurrent with quorum
    aggregator/
        weighted.py       # Weighted average
        voting.py         # Majority voting
        meta.py           # Meta-classifier stacking
    gate/
        model_armor.py    # Google Cloud Model Armor
    eval/
        runner.py         # Dataset loading & evaluation
        report.py         # Metrics & threshold recommendation
        calibration.py    # Platt scaling & isotonic regression
        batch.py          # Batch API adapters (stub)
```
