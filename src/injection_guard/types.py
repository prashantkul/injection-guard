"""All shared types for injection-guard. No internal imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Literal, runtime_checkable

__all__ = [
    "Action",
    "AggregatorType",
    "UnicodeSignals",
    "EncodingSignals",
    "StructuralSignals",
    "TokenSignals",
    "EntityMatch",
    "LinguisticSignals",
    "SignalVector",
    "PreprocessorOutput",
    "ClassifierResult",
    "BaseClassifier",
    "RouterConfig",
    "CascadeConfig",
    "ParallelConfig",
    "ThresholdConfig",
    "ModelArmorResult",
    "Decision",
    "EvalSample",
    "EvalMetrics",
]

# === Enums ===


class Action(str, Enum):
    """Decision action for a classified prompt."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class AggregatorType(str, Enum):
    """Supported aggregation strategies."""

    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    META_CLASSIFIER = "meta_classifier"


# === Preprocessor Types ===


@dataclass
class UnicodeSignals:
    """Signals from Stage 1: Unicode normalization analysis."""

    homoglyph_count: int = 0
    zero_width_count: int = 0
    bidi_override_count: int = 0
    normalization_edit_distance: int = 0
    script_mixing: bool = False
    suspicious_codepoints: list[str] = field(default_factory=list)


@dataclass
class EncodingSignals:
    """Signals from Stage 2: Encoding detection analysis."""

    encodings_found: list[str] = field(default_factory=list)
    decoded_payloads: list[str] = field(default_factory=list)
    encoding_density: float = 0.0
    nested_encoding: bool = False


@dataclass
class StructuralSignals:
    """Signals from Stage 3: Structural analysis."""

    chat_delimiters_found: list[str] = field(default_factory=list)
    xml_html_tags: list[str] = field(default_factory=list)
    instruction_boundary_patterns: list[str] = field(default_factory=list)
    separator_density: float = 0.0


@dataclass
class TokenSignals:
    """Signals from Stage 4: Token boundary analysis."""

    reconstructed_keywords: list[str] = field(default_factory=list)
    prompt_length_percentile: float = 0.5
    repetition_ratio: float = 0.0


@dataclass
class EntityMatch:
    """A single entity match from GLiNER semantic analysis."""

    text: str = ""
    label: str = ""
    score: float = 0.0


@dataclass
class LinguisticSignals:
    """Signals from Stage 5: GLiNER semantic entity detection."""

    injection_entities: list[EntityMatch] = field(default_factory=list)
    entity_types_found: list[str] = field(default_factory=list)
    max_entity_confidence: float = 0.0
    entity_count: int = 0


@dataclass
class SignalVector:
    """Combined signals from all preprocessor stages."""

    unicode: UnicodeSignals = field(default_factory=UnicodeSignals)
    encoding: EncodingSignals = field(default_factory=EncodingSignals)
    structural: StructuralSignals = field(default_factory=StructuralSignals)
    token: TokenSignals = field(default_factory=TokenSignals)
    linguistic: LinguisticSignals = field(default_factory=LinguisticSignals)


@dataclass
class PreprocessorOutput:
    """Complete output from the preprocessor pipeline."""

    normalized_prompt: str = ""
    original_prompt: str = ""
    decoded_payloads: list[str] = field(default_factory=list)
    signals: SignalVector = field(default_factory=SignalVector)
    risk_prior: float = 0.0


# === Classifier Types ===


@dataclass
class ClassifierResult:
    """Result from a single classifier."""

    score: float  # [0.0, 1.0] — 0=benign, 1=injection
    label: str  # "benign" | "injection"
    confidence: float = 1.0
    reasoning: str | None = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class BaseClassifier(Protocol):
    """Protocol that all classifiers must implement."""

    name: str
    latency_tier: Literal["fast", "medium", "slow"]
    weight: float

    async def classify(
        self, prompt: str, signals: SignalVector | None = None
    ) -> ClassifierResult: ...


# === Router Types ===


@dataclass
class RouterConfig:
    """Base router configuration."""

    timeout_ms: float = 500.0
    max_retries: int = 2
    backoff_base_ms: float = 100.0


@dataclass
class CascadeConfig(RouterConfig):
    """Cascade router configuration."""

    fast_confidence: float = 0.85
    escalate_on_high_risk_prior: bool = True
    risk_prior_escalation_threshold: float = 0.7


@dataclass
class ParallelConfig(RouterConfig):
    """Parallel router configuration."""

    quorum: int = 2


# === Threshold Types ===


@dataclass
class ThresholdConfig:
    """Threshold configuration for decision engine."""

    block_threshold: float = 0.85
    flag_threshold: float = 0.50
    preprocessor_block_threshold: float | None = None


# === Decision Type ===


@dataclass
class ModelArmorResult:
    """Result from Google Cloud Model Armor gate."""

    match_found: bool = False
    confidence_level: str | None = None
    pi_and_jailbreak: bool = False
    malicious_urls: list[str] = field(default_factory=list)
    sdp_findings: list[str] = field(default_factory=list)
    rai_findings: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Final classification decision with full audit trail."""

    action: Action
    ensemble_score: float
    model_scores: dict[str, ClassifierResult] = field(default_factory=dict)
    preprocessor: PreprocessorOutput = field(default_factory=PreprocessorOutput)
    model_armor: ModelArmorResult | None = None
    router_path: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    degraded: bool = False
    reasoning: str | None = None


# === Eval Types ===


@dataclass
class EvalSample:
    """A single evaluation sample."""

    prompt: str
    label: str  # "benign" | "injection"


@dataclass
class EvalMetrics:
    """Evaluation metrics for a classifier or ensemble."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    fpr: float = 0.0
    fnr: float = 0.0
