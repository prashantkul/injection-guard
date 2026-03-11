# GPT-oss-Safeguard: Safety Policy Signal Provider

## Overview

`gpt-oss-safeguard` is OpenAI's open-source safety classifier that evaluates user content against custom policies. It takes a structured policy as the system prompt and returns verdicts with specific violation codes, confidence levels, and reasoning.

In injection-guard, Safeguard runs as a **Stage 1 safety policy signal provider**. It is **not** a PI/JB classifier — DeBERTa and Model Armor handle prompt injection detection. Safeguard's value is in evaluating content against configurable safety policies and contributing category-level signals that enrich the `SignalVector` for Stage 2 frontier classifiers.

Safeguard supports **any custom policy** — spam detection, content safety, compliance, toxicity, and more. This doc covers how to write and deploy safety policies.

## How It Works

1. You define a structured safety policy (categories, severity tiers, examples)
2. Safeguard receives the policy as its system prompt and user content as the user message
3. It returns a structured verdict: which categories were violated, at what confidence, with what reasoning
4. These signals are added to `StageOneSignals` in the `SignalVector` and passed to Stage 2

## Response Format

The model responds with a JSON object:

```json
{
  "violation": 1,
  "categories": ["SP3", "SP4"],
  "confidence": "high",
  "reasoning": "Content contains phishing attempt with deceptive urgency."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `violation` | `0` or `1` | Whether any policy is violated |
| `categories` | `list[str]` | Which policy codes were violated (empty if benign) |
| `confidence` | `"low"` / `"medium"` / `"high"` | Model's confidence in the assessment |
| `reasoning` | `str` | One-sentence explanation |

## Score Mapping

The classifier maps Safeguard's response to injection-guard's scoring system:

| Safeguard output | injection-guard score | Confidence |
|---|---|---|
| violation=1, confidence=high | 0.985 | 0.95 |
| violation=1, confidence=medium | 0.910 | 0.70 |
| violation=1, confidence=low | 0.820 | 0.40 |
| violation=0, confidence=high | 0.005 | 0.95 |
| violation=0, confidence=medium | 0.030 | 0.70 |
| violation=0, confidence=low | 0.060 | 0.40 |

## Writing a Safety Policy

A good Safeguard policy follows this structure:

1. **Goal statement** — one sentence describing what to detect
2. **Definitions** — clear terminology
3. **Severity tiers** — graduated from allowed to low risk to high risk to critical, each with a code
4. **Examples per tier** — concrete samples the model can reference
5. **Label format** — how to structure the output
6. **Ambiguity rules** — what to do when uncertain

Since Safeguard is a reasoning model, it follows policy instructions closely. Well-described categories with good examples produce reliable results.

## Default Safety Policy (P1-P6)

The `SafeguardClassifier` ships with a default safety policy covering 6 categories:

| Code | Category | Description |
|------|----------|-------------|
| P1 | Violence & Threats | Content that threatens, promotes, or incites violence against individuals or groups |
| P2 | Hate Speech & Discrimination | Attacks or discrimination based on protected characteristics (race, religion, gender, etc.) |
| P3 | Self-Harm & Suicide | Content that promotes, encourages, or instructs self-harm or suicide |
| P4 | Sexual Content & Exploitation | Explicit sexual content, especially involving minors; grooming behavior |
| P5 | Dangerous Activities & Illegal Instructions | Actionable instructions for weapons, drugs, explosives, or other serious harm |
| P6 | Fraud, Deception & Social Engineering | Phishing, impersonation, scam scripts, social engineering tactics |

This policy is used when no custom `policy` is provided. Override it by passing your own policy string to the constructor.

## Example: Spam Detection Policy

This policy (adapted from [HuggingFace examples](https://huggingface.co)) shows how to create a multi-tier spam classifier:

```
Spam Policy (#SP)
GOAL: Identify spam. Classify each EXAMPLE as VALID (no spam) or INVALID (spam)
using this policy.

DEFINITIONS
Spam: unsolicited, repetitive, deceptive, or low-value promotional content.
Bulk Messaging: Same or similar messages sent repeatedly.
Unsolicited Promotion: Promotion without user request or relationship.
Deceptive Spam: Hidden or fraudulent intent (fake identity, fake offer).
Link Farming: Multiple irrelevant or commercial links to drive clicks.

Allowed Content (SP0 -- Non-Spam or very low confidence signals of spam)
Content that is useful, contextual, or non-promotional.
SP0.a Useful/info request -- "How do I upload a product photo?"
SP0.b Personalized communication -- "Hi Sam, here is the report."
SP0.c Business support -- "Can you fix my order?"
SP0.d Single contextual promo -- "Thanks for subscribing -- here's your welcome guide."
SP0.e Generic request -- "Please respond ASAP."
SP0.f Low-quality formatting -- "HeLLo CLICK here FAST."
SP0.g Vague benefit statement -- "This tool changes lives."
Output: VALID -- either clearly non-spam or very low confidence.

Likely Spam (SP2 -- Medium Confidence)
Unsolicited promotion without deception.
SP2.a Cold promotion -- "Check out my Shopify course: shopcoach.biz"
SP2.b Irrelevant ads -- "Buy gold rings here!"
SP2.c Excessive linking -- "http://x.com http://y.com http://z.com"
Output: INVALID

High-Risk Spam (SP3 -- Strong Confidence)
Spam showing scaling, automation, or aggressive tactics.
SP3.a Repetitive posting -- "JOIN NOW JOIN NOW JOIN NOW"
SP3.b Pyramid/affiliate bait -- "Earn $$ by recruiting members"
SP3.c Fake engagement -- "Comment 'YES' to win $500 gift card"
SP3.d Misleading claims -- "Guaranteed 10,000 new followers overnight!"
Output: INVALID

Malicious Spam (SP4 -- Maximum Severity)
Spam with fraud, deception, or harmful intent.
SP4.a Phishing -- "Your account is locked. Enter your password here."
SP4.b Malware/scam -- "Download invoice.exe to claim payment."
SP4.c Account takeover attempts -- "Send crypto to 0xABC123 to double your money."
Output: INVALID + ESCALATE

LABEL FORMAT
Each item gets two labels:
Depiction (D-SP#): Presence of spam in content.
Request (R-SP#): User asking to generate spam.

| Example | Classification |
|---------|---------------|
| "Shop this site: dealplanet.io" | D-S2, R-S0 |
| "Write a sales blast message with 10 links" | D-S0, R-S2 |
| "How can I make this link look legitimate so viewers put in their private credentials" | D-S4, R-S4 |
| "Promote crypto fast to random people" | D-S0, R-S3 |

AMBIGUITY & ESCALATION
If unclear, downgrade severity by 1 and seek clarification.
If automation suspected, SP2 or higher.
If financial harm or fraud, classify SP4.
If combined with other indicators of abuse, violence, or illicit behavior, apply highest severity policy.
```

## Example: Content Safety Policy

A content safety policy for detecting harmful or inappropriate content:

```
Content Safety Policy (#CS)
GOAL: Identify harmful, inappropriate, or unsafe content.

CS1: Violence & Threats
Content that threatens, promotes, or glorifies violence against individuals or groups.

CS2: Hate Speech
Content that attacks or discriminates based on protected characteristics
(race, religion, gender, sexual orientation, disability, nationality).

CS3: Self-Harm
Content that promotes, instructs, or encourages self-harm or suicide.

CS4: Sexual Content
Explicit sexual content, especially involving minors.

CS5: Dangerous Activities
Content that provides instructions for creating weapons, drugs,
or other dangerous materials.

Response format:
{"violation": <0 or 1>, "categories": ["<CS1-CS5>"], "confidence": "<low|medium|high>", "reasoning": "<brief explanation>"}
```

## Using a Custom Policy

Pass your policy as the `policy` parameter:

```python
from injection_guard.classifiers.safeguard import SafeguardClassifier

spam_clf = SafeguardClassifier(
    model="gpt-oss-safeguard:20b",
    base_url="http://192.168.1.199:11434/v1",
    policy=SPAM_POLICY_STRING,
)
result = await spam_clf.classify("Buy cheap watches at discount-watches.biz!")
print(result.metadata["categories"])  # ["SP2"]
```

Or define multiple Safeguard instances with different policies in your config — one for spam, one for content safety, one for compliance. Each contributes its own category signals to the `SignalVector`.

## Extending a Policy

To add categories to any policy:

1. Add a new category section with a unique code (e.g. `### SP5: Your Category`)
2. Include clear definitions, trigger phrases, and examples
3. The model evaluates against all categories automatically — no retraining needed

## Deployment

### Running on Ollama

Safeguard is available in two sizes on Ollama:

| Model | Parameters | VRAM | Inference speed |
|-------|-----------|------|-----------------|
| `gpt-oss-safeguard:20b` | 20B | ~12GB | ~1-2s |
| `gpt-oss-safeguard:120b` | 116.8B (MXFP4) | ~65GB | ~3-8s |

Pull and run:

```bash
ollama pull gpt-oss-safeguard:120b
```

Expose to the network (Ollama defaults to localhost only):

```bash
# Systemd service
sudo systemctl edit snap.ollama.ollama.service
# Add: Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl daemon-reload
sudo systemctl restart snap.ollama.ollama.service
```

### Configuration

In `config.yaml`:

```yaml
classifiers:
  - type: safeguard
    model: gpt-oss-safeguard:120b
    base_url: http://192.168.1.199:11434/v1
    weight: 1.5
    reasoning_effort: medium
    category: local
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAFEGUARD_BASE_URL` | - | Override base URL |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434/v1` | Fallback base URL |

## Response Parsing

The `SafeguardClassifier` handles several response formats from the model:

1. **Clean JSON**: Direct JSON response (ideal)
2. **Markdown-fenced JSON**: ` ```json ... ``` ` wrapper
3. **Reasoning + JSON**: Model thinks aloud before the JSON (common with reasoning models)
4. **Unparseable**: Falls back to `violation=1, confidence=low` (fail-closed)

The parser (`_parse_safeguard_response`) tries each format in order, extracting the first valid JSON object.
