"""EvalReport — evaluation metrics and threshold recommendation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from injection_guard.types import (
    Action,
    ClassifierResult,
    Decision,
    EvalMetrics,
    ThresholdConfig,
)

__all__ = ["EvalReport"]


class EvalReport:
    """Evaluation report computed from prediction/ground-truth pairs.

    Args:
        predictions: List of ``(Decision, ground_truth_label)`` pairs where
            ``ground_truth_label`` is ``"injection"`` or ``"benign"``.
    """

    def __init__(self, predictions: list[tuple[Decision, str]]) -> None:
        self._predictions = predictions
        self._scores: list[float] = [d.ensemble_score for d, _ in predictions]
        self._y_true: list[int] = [1 if lab == "injection" else 0 for _, lab in predictions]
        self._y_pred: list[int] = [
            1 if d.action in (Action.BLOCK, Action.FLAG) else 0 for d, _ in predictions
        ]

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def confusion_matrix(self) -> dict[str, int]:
        """Compute the confusion matrix.

        Returns:
            Dict with keys ``TP``, ``FP``, ``TN``, ``FN``.
        """
        tp = fp = tn = fn = 0
        for yt, yp in zip(self._y_true, self._y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 0 and yp == 0:
                tn += 1
            else:
                fn += 1
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    # ------------------------------------------------------------------
    # Precision / Recall / F1
    # ------------------------------------------------------------------

    def precision_recall_f1(self) -> EvalMetrics:
        """Compute precision, recall, F1, FPR, and FNR.

        Returns:
            An ``EvalMetrics`` instance with all fields populated.
        """
        cm = self.confusion_matrix()
        tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return EvalMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=self.roc_auc(),
            confusion_matrix=[[tn, fp], [fn, tp]],
            fpr=fpr,
            fnr=fnr,
        )

    # ------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------

    def roc_auc(self) -> float:
        """Compute ROC-AUC score.

        Uses scikit-learn when available; otherwise computes the
        Wilcoxon-Mann-Whitney statistic (equivalent to AUC).

        Returns:
            The ROC-AUC value, or 0.0 if undefined.
        """
        try:
            from sklearn.metrics import roc_auc_score
            import numpy as np

            return float(roc_auc_score(np.array(self._y_true), np.array(self._scores)))
        except (ImportError, ValueError):
            return self._manual_roc_auc()

    def _manual_roc_auc(self) -> float:
        """Wilcoxon-Mann-Whitney AUC computation."""
        pos_scores = [s for s, y in zip(self._scores, self._y_true) if y == 1]
        neg_scores = [s for s, y in zip(self._scores, self._y_true) if y == 0]

        if not pos_scores or not neg_scores:
            return 0.0

        count = 0
        ties = 0
        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    count += 1
                elif p == n:
                    ties += 1

        return (count + 0.5 * ties) / (len(pos_scores) * len(neg_scores))

    # ------------------------------------------------------------------
    # Score distribution
    # ------------------------------------------------------------------

    def score_distribution(self, bins: int = 10) -> dict[str, list[float] | list[int]]:
        """Histogram of ensemble scores split by ground-truth label.

        Args:
            bins: Number of histogram bins.

        Returns:
            Dict with ``bin_edges``, ``injection_counts``, and
            ``benign_counts``.
        """
        edges = [i / bins for i in range(bins + 1)]
        inj_counts = [0] * bins
        ben_counts = [0] * bins

        for score, yt in zip(self._scores, self._y_true):
            idx = min(int(score * bins), bins - 1)
            if yt == 1:
                inj_counts[idx] += 1
            else:
                ben_counts[idx] += 1

        return {
            "bin_edges": edges,
            "injection_counts": inj_counts,
            "benign_counts": ben_counts,
        }

    # ------------------------------------------------------------------
    # Threshold recommendation
    # ------------------------------------------------------------------

    def recommend_thresholds(
        self,
        target_fpr: float | None = None,
        target_fnr: float | None = None,
    ) -> ThresholdConfig:
        """Find optimal thresholds to meet a target FPR or FNR.

        Sweeps over possible threshold values and picks the best operating
        point. If neither target is given, optimises for F1.

        Args:
            target_fpr: Desired maximum false-positive rate.
            target_fnr: Desired maximum false-negative rate.

        Returns:
            A ``ThresholdConfig`` with recommended ``block_threshold``
            and ``flag_threshold``.
        """
        # Build sorted unique thresholds to test
        thresholds = sorted(set(self._scores))
        if not thresholds:
            return ThresholdConfig()

        # Add boundary values
        candidates = [0.0] + thresholds + [1.0]

        best_block = 0.85
        best_flag = 0.50
        best_metric = -1.0

        for t in candidates:
            tp = fp = tn = fn = 0
            for score, yt in zip(self._scores, self._y_true):
                pred = 1 if score >= t else 0
                if yt == 1 and pred == 1:
                    tp += 1
                elif yt == 0 and pred == 1:
                    fp += 1
                elif yt == 0 and pred == 0:
                    tn += 1
                else:
                    fn += 1

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            if target_fpr is not None:
                if fpr <= target_fpr and recall > best_metric:
                    best_metric = recall
                    best_block = t
            elif target_fnr is not None:
                if fnr <= target_fnr and (1.0 - fpr) > best_metric:
                    best_metric = 1.0 - fpr
                    best_block = t
            else:
                if f1 > best_metric:
                    best_metric = f1
                    best_block = t

        # Flag threshold is set lower than block
        best_flag = max(0.0, best_block - 0.20)

        return ThresholdConfig(
            block_threshold=best_block,
            flag_threshold=best_flag,
        )

    # ------------------------------------------------------------------
    # Per-model diagnostics
    # ------------------------------------------------------------------

    def per_model_diagnostics(self) -> dict[str, EvalMetrics]:
        """Compute per-model metrics using individual model scores.

        Returns:
            Dict mapping model names to ``EvalMetrics``.
        """
        # Collect all model names
        model_names: set[str] = set()
        for decision, _ in self._predictions:
            model_names.update(decision.model_scores.keys())

        result: dict[str, EvalMetrics] = {}
        for name in sorted(model_names):
            model_scores: list[float] = []
            model_y_true: list[int] = []
            model_y_pred: list[int] = []

            for decision, label in self._predictions:
                if name not in decision.model_scores:
                    continue
                cr: ClassifierResult = decision.model_scores[name]
                model_scores.append(cr.score)
                yt = 1 if label == "injection" else 0
                yp = 1 if cr.label == "injection" else 0
                model_y_true.append(yt)
                model_y_pred.append(yp)

            if not model_scores:
                continue

            tp = fp = tn = fn = 0
            for yt, yp in zip(model_y_true, model_y_pred):
                if yt == 1 and yp == 1:
                    tp += 1
                elif yt == 0 and yp == 1:
                    fp += 1
                elif yt == 0 and yp == 0:
                    tn += 1
                else:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # ROC-AUC per model
            roc = 0.0
            pos = [s for s, y in zip(model_scores, model_y_true) if y == 1]
            neg = [s for s, y in zip(model_scores, model_y_true) if y == 0]
            if pos and neg:
                count = 0
                ties = 0
                for p in pos:
                    for n in neg:
                        if p > n:
                            count += 1
                        elif p == n:
                            ties += 1
                roc = (count + 0.5 * ties) / (len(pos) * len(neg))

            result[name] = EvalMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc,
                confusion_matrix=[[tn, fp], [fn, tp]],
                fpr=fpr,
                fnr=fnr,
            )

        return result

    # ------------------------------------------------------------------
    # Calibration stubs
    # ------------------------------------------------------------------

    def calibration_curves(self, n_bins: int = 10) -> dict[str, list[float]]:
        """Compute calibration curve data (reliability diagram).

        Args:
            n_bins: Number of calibration bins.

        Returns:
            Dict with ``mean_predicted``, ``fraction_positive``, and
            ``bin_counts``.

        Note:
            Plotting requires matplotlib (not imported here).
        """
        bin_sums = [0.0] * n_bins
        bin_true = [0.0] * n_bins
        bin_counts = [0] * n_bins

        for score, yt in zip(self._scores, self._y_true):
            idx = min(int(score * n_bins), n_bins - 1)
            bin_sums[idx] += score
            bin_true[idx] += yt
            bin_counts[idx] += 1

        mean_predicted: list[float] = []
        fraction_positive: list[float] = []
        counts_out: list[float] = []

        for i in range(n_bins):
            if bin_counts[i] > 0:
                mean_predicted.append(bin_sums[i] / bin_counts[i])
                fraction_positive.append(bin_true[i] / bin_counts[i])
                counts_out.append(float(bin_counts[i]))
            else:
                mean_predicted.append((i + 0.5) / n_bins)
                fraction_positive.append(0.0)
                counts_out.append(0.0)

        return {
            "mean_predicted": mean_predicted,
            "fraction_positive": fraction_positive,
            "bin_counts": counts_out,
        }

    def fit_calibration(self) -> dict[str, list[float]]:
        """Fit calibration models and return calibrated scores.

        Returns:
            Dict with ``platt_scores`` and ``isotonic_scores`` keys
            containing calibrated probability estimates. Returns empty
            lists if calibration fails.

        Note:
            Requires scikit-learn for full functionality. Falls back to
            manual implementations from ``calibration.py`` otherwise.
        """
        from injection_guard.eval.calibration import PlattScaler, IsotonicCalibrator

        result: dict[str, list[float]] = {
            "platt_scores": [],
            "isotonic_scores": [],
        }

        try:
            platt = PlattScaler().fit(self._scores, self._y_true)
            result["platt_scores"] = platt.transform(self._scores)
        except Exception:
            pass

        try:
            iso = IsotonicCalibrator().fit(self._scores, self._y_true)
            result["isotonic_scores"] = iso.transform(self._scores)
        except Exception:
            pass

        return result
