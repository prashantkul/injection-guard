# Evaluation Results

## Model Armor — Qualifire Dataset (200 samples)

Balanced sample: 100 injection, 100 benign. Seed=42.
Dataset: `qualifire/prompt-injections-benchmark` (test split).

| Template | TP | FN | TN | FP | Precision | Recall | F1 | Accuracy | Avg Latency |
|----------|---:|---:|---:|---:|----------:|-------:|-----:|---------:|------------:|
| MA High | 18 | 82 | 99 | 1 | 0.947 | 0.180 | 0.303 | 58.5% | 180ms |
| MA Medium | 45 | 55 | 96 | 4 | 0.918 | 0.450 | 0.604 | 70.5% | 676ms |
| MA Low | 56 | 44 | 93 | 7 | 0.889 | 0.560 | 0.687 | 74.5% | 799ms |

### Observations

- **MA High**: Near-perfect precision (94.7%) but very low recall (18%). Only catches the most obvious injection patterns. Best suited as a zero-false-positive pre-gate.
- **MA Medium**: Balanced trade-off. Catches ~45% of injections with 4% false positive rate.
- **MA Low**: Best overall accuracy (74.5%) but still misses 44% of injections. 7% false positive rate.
- All templates have very low false positive rates (1-7%), making them safe as pre-filters.
- Model Armor is fast (180-800ms per request) compared to LLM classifiers.
- Model Armor works best as a **pre-gate** before the ensemble classifier, not as a standalone detector.

## API Classifiers — Qualifire Dataset (200 samples)

Balanced sample: 100 injection, 100 benign. Seed=42.
Anthropic via Message Batches API (50% cost savings). Others via async parallel.

| Model | Reasoning | TP | FN | TN | FP | Precision | Recall | F1 | Accuracy |
|-------|-----------|---:|---:|---:|---:|----------:|-------:|-----:|---------:|
| Anthropic (claude-opus-4.6) | — | 80 | 20 | 87 | 13 | 0.860 | 0.800 | 0.829 | 83.5% |
| Anthropic (claude-sonnet-4.6) | — | 89 | 11 | 76 | 24 | 0.788 | 0.890 | 0.836 | 82.5% |
| OpenAI (gpt-5) | high | 94 | 6 | 70 | 30 | 0.758 | 0.940 | 0.839 | 82.0% |
| OpenAI (gpt-5) | medium | 74 | 26 | 83 | 17 | 0.813 | 0.740 | 0.775 | 78.5% |
| OpenAI (gpt-5-mini) | high | 97 | 3 | 60 | 40 | 0.708 | 0.970 | 0.819 | 78.5% |
| OpenAI (gpt-5-mini) | medium | 83 | 17 | 75 | 25 | 0.769 | 0.830 | 0.798 | 79.0% |
| Gemini (3.1-pro-preview) | — | 94 | 6 | 67 | 33 | 0.740 | 0.940 | 0.828 | 80.5% |
| Gemini (3-flash-preview) | — | 97 | 3 | 63 | 37 | 0.724 | 0.970 | 0.829 | 80.0% |

### Reasoning Effort Impact (OpenAI)

| Model | Effort | Accuracy | Recall | Precision | F1 |
|-------|--------|----------|--------|-----------|-----|
| gpt-5 | medium | 78.5% | 0.740 | 0.813 | 0.775 |
| gpt-5 | **high** | **82.0%** | **0.940** | 0.758 | **0.839** |
| gpt-5-mini | medium | 79.0% | 0.830 | 0.769 | 0.798 |
| gpt-5-mini | **high** | 78.5% | **0.970** | 0.708 | 0.819 |

High reasoning dramatically boosts recall (74→94% for gpt-5, 83→97% for mini) at the cost of precision.
gpt-5 + high reasoning gains +3.5% accuracy; gpt-5-mini stays flat because the recall gain is offset by more false positives.

## Open-Weight Models — Qualifire Dataset (200 samples)

HuggingFace classification models served via litguard (LitServe) on DGX.
Safeguard models served via Ollama on DGX.

| Model | TP | FN | TN | FP | Precision | Recall | F1 | Accuracy |
|-------|---:|---:|---:|---:|----------:|-------:|-----:|---------:|
| protectai/deberta-v3-base-prompt-injection-v2 | 65 | 35 | 74 | 26 | 0.714 | 0.650 | 0.681 | 69.5% |
| deepset/deberta-v3-base-injection | 99 | 1 | 31 | 69 | 0.589 | 0.990 | 0.739 | 65.0% |
| Safeguard 20B (Ollama) | 23 | 77 | 98 | 2 | 0.920 | 0.230 | 0.368 | 60.5% |
| Safeguard 120B (Ollama) | 19 | 81 | 96 | 4 | 0.826 | 0.190 | 0.309 | 57.5% |

### Observations

**API Classifiers:**
- **Anthropic claude-opus-4.6**: Best accuracy (83.5%) with strong precision (86%) and recall (80%). Best overall balance.
- **Anthropic claude-sonnet-4.6**: Highest F1 (0.836) with best recall among Anthropic (89%) but more false positives (24%).
- **OpenAI gpt-5 + high reasoning**: Matches Anthropic at 82.0%. Reasoning tokens transform OpenAI from worst (78.5%) to competitive. Very high recall (94%).
- **Gemini 3.1-pro-preview**: Strong at 80.5%, very high recall (94%) but lower precision (74%).
- **Gemini 3-flash-preview**: Similar to Pro at 80.0%. Near-perfect recall (97%) but highest false positive rate (37%).
- High reasoning effort is critical for OpenAI — without it, gpt-5 underperforms significantly.

**Open-Weight Models:**
- **protectai/deberta-v3-base-prompt-injection-v2**: Best open-weight accuracy (69.5%). Balanced precision/recall. Purpose-built for prompt injection.
- **deepset/deberta-v3-base-injection**: Near-perfect recall (99%) but very high false positive rate (69%). Best as a pre-filter — catches nearly everything but flags too many benign prompts.
- **Safeguard 20B/120B**: Very high precision (83-92%) but extremely low recall (19-23%). Useful only as zero-FP pre-filters. 120B paradoxically worse than 20B.
- Open-weight models are 10-100x faster than API classifiers (~100ms vs 1-10s).

**Recommended Ensemble Strategy:**
1. **Fast pre-filter**: deepset/deberta (99% recall, ~100ms) — catch obvious injections instantly
2. **Primary classifier**: Anthropic claude-opus-4.6 or OpenAI gpt-5 + high reasoning (~82-84% accuracy)
3. **High-precision tiebreaker**: Safeguard 20B (92% precision) — confirm uncertain cases
