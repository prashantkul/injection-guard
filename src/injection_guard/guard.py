"""InjectionGuard — main orchestrator for prompt injection detection."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

import logging

from injection_guard.types import (
    Action,
    AggregatorType,
    BaseClassifier,
    ClassifierResult,
    Decision,
    ModelArmorResult,
    PreprocessorOutput,
    RouteResult,
    SignalVector,
    StageOneSignals,
    ThresholdConfig,
)

logger = logging.getLogger(__name__)
from injection_guard.preprocessor.pipeline import Preprocessor
from injection_guard.engine import ThresholdEngine
from injection_guard.aggregator import get_aggregator

__all__ = ["InjectionGuard"]


class InjectionGuard:
    """Ensemble prompt injection detection with pluggable classifiers.

    Orchestrates the full pipeline: preprocessor → model armor gate →
    ensemble router → aggregator → threshold engine → decision.

    Args:
        classifiers: List of classifiers conforming to BaseClassifier protocol.
        router: A CascadeRouter or ParallelRouter instance.
        model_armor: Optional Model Armor gate for GCP pre-screening.
        gliner_model: HuggingFace model name for GLiNER stage.
        gliner_device: Device for GLiNER inference ("cuda" or "cpu").
        thresholds: Dict with "block" and "flag" threshold values.
        preprocessor_block_threshold: If set, preprocessor blocks above this.
        aggregator: Aggregation strategy name.
        meta_classifier_path: Path to trained meta-classifier model.
        log_prompts: Whether to log prompts (default False for privacy).
    """

    def __init__(
        self,
        classifiers: list[BaseClassifier],
        router: object,
        *,
        model_armor: object | None = None,
        gliner_model: str = "urchade/gliner_base",
        gliner_device: str = "cpu",
        thresholds: dict[str, float] | None = None,
        preprocessor_block_threshold: float | None = None,
        aggregator: str = "weighted_average",
        meta_classifier_path: str | None = None,
        log_prompts: bool = False,
        dotenv_path: str | None = None,
    ) -> None:
        # Load .env file so API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
        # are available via os.environ without manual export.
        load_dotenv(dotenv_path=dotenv_path, override=False)

        thresholds = thresholds or {"block": 0.85, "flag": 0.50}

        # Separate Stage 1 (signal enrichment) from Stage 2 (frontier ensemble)
        self._stage1_classifiers: list[BaseClassifier] = []
        self._classifiers: list[BaseClassifier] = []
        for clf in classifiers:
            if self._is_stage1(clf):
                self._stage1_classifiers.append(clf)
            else:
                self._classifiers.append(clf)

        self._router = router
        self._model_armor = model_armor
        self._log_prompts = log_prompts

        self._preprocessor = Preprocessor(gliner_model=gliner_model)
        self._engine = ThresholdEngine(
            ThresholdConfig(
                block_threshold=thresholds.get("block", 0.85),
                flag_threshold=thresholds.get("flag", 0.50),
                preprocessor_block_threshold=preprocessor_block_threshold,
            )
        )

        agg_type = AggregatorType(aggregator)
        self._aggregator = get_aggregator(agg_type, meta_model_path=meta_classifier_path)

    @classmethod
    def from_config(cls, config: str | Path | dict[str, Any]) -> InjectionGuard:
        """Create an InjectionGuard instance from a YAML file or config dict.

        Args:
            config: Path to a YAML config file, or a pre-parsed config dict.

        Returns:
            A fully configured InjectionGuard instance.
        """
        from injection_guard.config import load_config, build_from_config

        if isinstance(config, (str, Path)):
            raw = load_config(config)
        else:
            raw = config
        kwargs = build_from_config(raw)
        return cls(**kwargs)

    @staticmethod
    def _is_stage1(clf: BaseClassifier) -> bool:
        """Check if a classifier is a Stage 1 signal enricher."""
        cls_name = type(clf).__name__
        return cls_name in ("SafeguardClassifier", "HFCompatClassifier")

    async def _run_stage1(
        self, prompt: str, signals: SignalVector,
    ) -> None:
        """Run Stage 1 classifiers and enrich SignalVector in-place.

        Stage 1 never blocks — it only enriches signals for Stage 2.
        """
        for clf in self._stage1_classifiers:
            try:
                result = await asyncio.wait_for(clf.classify(prompt, signals), timeout=10.0)
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning("Stage 1 classifier %s failed: %s", clf.name, exc)
                continue

            if result.metadata.get("error"):
                logger.warning("Stage 1 classifier %s returned error: %s", clf.name, result.metadata["error"])
                continue

            cls_name = type(clf).__name__
            if cls_name == "SafeguardClassifier":
                signals.stage_one.safeguard_violation = bool(result.metadata.get("violation"))
                signals.stage_one.safeguard_confidence = result.metadata.get("safeguard_confidence")
                signals.stage_one.safeguard_categories = result.metadata.get("categories", [])
                signals.stage_one.safeguard_reasoning = result.reasoning
            elif cls_name == "HFCompatClassifier":
                signals.stage_one.deberta_score = result.score
                signals.stage_one.deberta_label = result.label
                signals.stage_one.deberta_confidence = result.confidence

    async def classify(self, prompt: str) -> Decision:
        """Classify a prompt for injection attempts.

        Args:
            prompt: The raw user prompt to classify.

        Returns:
            A Decision object with action, scores, and full audit trail.
        """
        start = time.perf_counter()

        # 1. Preprocess (Stages 1-5 including GLiNER)
        prep = self._preprocessor.process(prompt)

        # 2. Check preprocessor block threshold
        if self._engine.preprocessor_blocks(prep.risk_prior):
            return Decision(
                action=Action.BLOCK,
                ensemble_score=prep.risk_prior,
                preprocessor=prep,
                router_path=["preprocessor-block"],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 3. Model Armor gate (if configured)
        ma_result: ModelArmorResult | None = None
        if self._model_armor is not None:
            ma_result = await self._model_armor.screen(prep.normalized_prompt)
            if ma_result.pi_and_jailbreak and ma_result.confidence_level == "HIGH":
                return Decision(
                    action=Action.BLOCK,
                    ensemble_score=1.0,
                    preprocessor=prep,
                    model_armor=ma_result,
                    router_path=["model-armor-block"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            if ma_result.pi_and_jailbreak:
                # MEDIUM confidence — boost risk_prior
                prep.risk_prior = min(1.0, prep.risk_prior + 0.3)

        # 4. Stage 1: signal enrichment (never blocks)
        if self._stage1_classifiers:
            await self._run_stage1(prep.normalized_prompt, prep.signals)

        # 5. Route to Stage 2 classifiers
        route_result: RouteResult = await self._router.route(
            classifiers=self._classifiers,
            prompt=prep.normalized_prompt,
            signals=prep.signals,
            risk_prior=prep.risk_prior,
        )
        route_results = route_result.results

        # 6. Aggregate scores
        clf_map = {c.name: c for c in self._classifiers}
        if route_results:
            agg_pairs = []
            for name, result in route_results:
                clf = clf_map.get(name)
                if clf is not None:
                    agg_pairs.append((clf, result))

            ensemble_score, ensemble_label = self._aggregator.aggregate(agg_pairs)
        else:
            ensemble_score = prep.risk_prior
            ensemble_label = "injection" if prep.risk_prior >= 0.5 else "benign"

        # 7. Threshold → action
        action = self._engine.decide(ensemble_score)

        # Build model_scores dict
        model_scores = {name: result for name, result in route_results}

        # Find reasoning from highest-weight API model and check for errors
        reasoning = None
        degraded = False
        for name, result in route_results:
            if result.metadata.get("error"):
                degraded = True
            if result.reasoning and reasoning is None:
                reasoning = result.reasoning

        # Fail closed: if quorum was not met, block the prompt.
        # A degraded ensemble without sufficient classifier responses
        # cannot be trusted to allow prompts through.
        if not route_result.quorum_met:
            action = Action.BLOCK
            degraded = True

        return Decision(
            action=action,
            ensemble_score=ensemble_score,
            model_scores=model_scores,
            preprocessor=prep,
            model_armor=ma_result,
            router_path=[name for name, _ in route_results],
            latency_ms=(time.perf_counter() - start) * 1000,
            degraded=degraded,
            reasoning=reasoning,
        )

    def classify_sync(self, prompt: str) -> Decision:
        """Synchronous wrapper for classify.

        Args:
            prompt: The raw user prompt to classify.

        Returns:
            A Decision object with action, scores, and full audit trail.
        """
        return asyncio.run(self.classify(prompt))

    async def classify_batch(self, prompts: list[str]) -> list[Decision]:
        """Classify multiple prompts concurrently.

        Args:
            prompts: List of raw user prompts.

        Returns:
            List of Decision objects, one per prompt.
        """
        return await asyncio.gather(*(self.classify(p) for p in prompts))

    def update_thresholds(
        self,
        block: float | None = None,
        flag: float | None = None,
        preprocessor_block: float | None = None,
    ) -> None:
        """Update classification thresholds at runtime.

        Args:
            block: New block threshold.
            flag: New flag threshold.
            preprocessor_block: New preprocessor block threshold.
        """
        self._engine.update_thresholds(
            block=block, flag=flag, preprocessor_block=preprocessor_block
        )

    def update_preprocessor_block_threshold(self, threshold: float | None) -> None:
        """Update the preprocessor block threshold.

        Args:
            threshold: New threshold value, or None to disable.
        """
        self._engine.update_thresholds(preprocessor_block=threshold)
