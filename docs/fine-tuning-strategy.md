# Fine-Tuning Open-Weight Models for Prompt Injection Detection

## The Gap

Current eval results on the Qualifire dataset (200 balanced samples):

| Model | Accuracy | Recall | Precision | F1 | Latency |
|-------|----------|--------|-----------|----|---------|
| deepset/deberta-v3-base-injection | 65.0% | 99% | 59% | 74% | ~100ms |
| protectai/deberta-v3-base-prompt-injection-v2 | 69.5% | 65% | 71% | 68% | ~100ms |
| Anthropic claude-opus-4.6 (API) | 83.5% | — | — | — | 1-10s |

The open-weight models sit 15-20 points below the best API classifier. Fine-tuning can close this gap while keeping local inference at ~100ms -- a 10-100x latency advantage over API calls.

The deepset model shows the classic "flag everything" failure mode (99% recall, 59% precision). The protectai model is balanced but mediocre on both axes. A well-tuned model should hit 80%+ accuracy with 95%+ recall.

## Dataset Strategy

### Primary Sources

1. **Qualifire dataset** -- use as eval holdout, not training data. It's too small (200 samples) to train on effectively.
2. **Existing public datasets:**
   - [Lakera gandalf](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) -- real user attacks
   - [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) -- the original training set
   - [JasperLS prompt-injections](https://huggingface.co/datasets/JasperLS/prompt-injections) -- expanded variant
   - [hackaprompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) -- competition submissions
3. **Synthetic data from frontier models** -- see Distillation section below.

### Augmentation Techniques

Prompt injection payloads have specific augmentation opportunities:

| Technique | Example | Purpose |
|-----------|---------|---------|
| Paraphrasing | "ignore previous" -> "disregard prior" | Synonym robustness |
| Encoding injection | Base64, ROT13, hex encoding of payloads | Catch obfuscated attacks |
| Language mixing | English instruction + Mandarin payload | Multilingual evasion |
| Delimiter manipulation | `"""`, `---`, `<system>` wrapping | Structural attacks |
| Nested injection | Benign outer + malicious inner prompt | Depth robustness |
| Case/whitespace variation | `IGNORE ALL`, `i g n o r e` | Surface-level evasion |

For negative samples (benign inputs), collect real-world user prompts from public datasets (ShareGPT, LMSYS-Chat-1M) to avoid the model learning superficial patterns like "any long prompt is an attack."

### Target Dataset Composition

- 10K-50K samples minimum
- 60/40 benign/malicious split (slight imbalance toward benign reflects production distribution)
- At least 20% of attack samples should use augmented/obfuscated payloads
- Stratify by attack category: instruction override, role hijack, data exfiltration, jailbreak

## Metrics and Optimization Targets

For security applications, **missing an attack is worse than a false alarm**. Optimize accordingly:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall | >= 95% | Hard floor. A missed injection is a security breach. |
| F1 | >= 85% | Primary optimization metric. Balances recall with usability. |
| Precision | >= 75% | Soft target. Below this, false positives degrade UX. |
| Accuracy | >= 82% | Sanity check. Should follow from recall + precision targets. |

### Threshold Tuning

Don't use 0.5 as the classification threshold. After training:

1. Compute precision-recall curve on validation set.
2. Find the threshold where recall = 0.95.
3. Check if precision at that threshold is acceptable (>= 0.70).
4. If not, the model needs more/better training data -- not a lower threshold.

Store the optimal threshold in model config so `HFCompatClassifier` uses it at inference time.

## Fine-Tuning Approach

### Base Model Selection

Start with `microsoft/deberta-v3-base` (86M params). The `-large` variant (304M params) gives ~2-3% accuracy gain but 3x inference cost. Base is the better tradeoff for a local classifier in an ensemble.

### Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deberta-injection-ft",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=4,
    report_to="wandb",
)
```

### LoRA for Efficiency

Full fine-tuning works well for DeBERTa-base but LoRA reduces memory and training time significantly if you're iterating fast or using `-large`:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
# Trainable params: ~300K vs 86M full model
```

Merge LoRA weights back into the base model before deployment to avoid inference overhead.

### Handling Class Imbalance

If your dataset is imbalanced, use class weights in the loss function:

```python
from torch import tensor

# Compute from training set distribution
class_weights = tensor([1.0, 2.5])  # [benign_weight, malicious_weight]

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

Alternatively, use focal loss to down-weight easy negatives. This helps when the model quickly learns obvious benign prompts but struggles with subtle attacks.

## Distillation from Frontier Models

The core idea: use Claude/GPT-4 as teachers to label a large unlabeled corpus, then train DeBERTa on those labels. This is the highest-leverage technique for closing the accuracy gap.

### Pipeline

1. **Collect unlabeled prompts** -- scrape public prompt datasets, generate synthetic prompts, mix benign and attack samples.
2. **Label with frontier model** -- use the injection-guard API classifier (Claude) with chain-of-thought prompting to produce:
   - Binary label (injection / benign)
   - Confidence score (0.0 - 1.0)
   - Attack category (if injection)
3. **Filter by confidence** -- only keep samples where teacher confidence >= 0.85. Discard ambiguous samples.
4. **Soft label training** -- instead of hard 0/1 labels, use the teacher's probability distribution as the target:

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_probs, temperature=3.0):
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature ** 2)
```

5. **Combine with hard labels** -- use a weighted sum of distillation loss and standard cross-entropy on the labeled subset:

```
total_loss = alpha * distillation_loss + (1 - alpha) * ce_loss
```

Start with `alpha=0.7` (favor teacher signal).

### Cost Considerations

Labeling 50K samples with Claude API at ~500 tokens/sample:
- Input: ~25M tokens at $15/MTok = ~$375
- Output: ~5M tokens at $75/MTok = ~$375
- Total: ~$750

This is a one-time cost that produces a reusable training dataset. Compare to ongoing API classifier costs in production.

## Evaluation Protocol

### Holdout Strategy

- **Train/val/test split**: 80/10/10, stratified by attack category.
- **Never touch the test set during development.** Use validation set for hyperparameter tuning.
- **Qualifire dataset**: Reserved as an independent out-of-distribution test set.

### Adversarial Evaluation

Maintain a separate adversarial test set with:
- Novel attack patterns not in training data
- Attacks generated after the training data cutoff
- Red-team submissions from internal testing
- Edge cases: very short prompts, prompts with code blocks, multilingual prompts

### Continuous Evaluation

Run eval weekly against:
1. Static test sets (regression check)
2. New attack samples from threat feeds (generalization check)
3. Production false positive samples (precision monitoring)

Track metrics over time. A fine-tuned model should be retrained quarterly or when recall drops below the 95% threshold on new attack samples.

### Cross-Validation

For final model selection, run 5-fold stratified cross-validation on the full training set. Report mean and std of F1 and recall. If recall std > 3%, the model is unstable -- investigate per-fold failures.

## Integration with injection-guard

Fine-tuned models plug directly into the existing ensemble via `HFCompatClassifier`.

### Model Packaging

After training, export the model with its optimal threshold:

```python
# Save model + tokenizer
model.save_pretrained("./injection-guard-deberta-v1")
tokenizer.save_pretrained("./injection-guard-deberta-v1")

# Save threshold in config
import json
config = {"optimal_threshold": 0.42, "model_version": "v1.0", "trained_on": "2026-03-10"}
with open("./injection-guard-deberta-v1/injection_guard_config.json", "w") as f:
    json.dump(config, f)
```

### Classifier Configuration

In the injection-guard config, register the fine-tuned model:

```python
from injection_guard.types import ClassifierConfig

ft_classifier = ClassifierConfig(
    name="injection-guard-deberta-v1",
    kind="hf",
    model_id="./models/injection-guard-deberta-v1",  # local path or HF hub ID
    threshold=0.42,  # from threshold tuning
    weight=1.5,  # higher weight in ensemble -- this is your best local model
)
```

The model runs through `HFCompatClassifier`, which handles tokenization and inference. No code changes needed in the classifier layer.

### Ensemble Strategy

With a fine-tuned local model, the recommended ensemble shifts:

| Classifier | Role | Weight | Latency |
|------------|------|--------|---------|
| Fine-tuned DeBERTa | Primary local | 1.5 | ~100ms |
| deepset/deberta (original) | Diversity signal | 0.5 | ~100ms |
| Claude API | High-accuracy fallback | 2.0 | 1-10s |

Use the router's tiered strategy: run both local models first (~100ms). If they agree, skip the API call entirely. If they disagree, escalate to Claude for the final verdict. This keeps median latency low while maintaining high accuracy on ambiguous inputs.

### litguard Integration

For ONNX-optimized inference, convert the fine-tuned model:

```bash
optimum-cli export onnx --model ./injection-guard-deberta-v1 ./injection-guard-deberta-v1-onnx
```

Then use via litguard for ~50ms inference instead of ~100ms with PyTorch.
