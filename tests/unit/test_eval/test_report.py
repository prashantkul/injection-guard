"""Tests for EvalReport metrics."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from injection_guard.eval.report import EvalReport
from injection_guard.types import Action, ClassifierResult, Decision


def _make_decision(action: Action, score: float, model_scores: dict | None = None) -> Decision:
    """Helper to create a Decision for tests."""
    return Decision(
        action=action,
        ensemble_score=score,
        model_scores=model_scores or {},
    )


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_perfect_predictions(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.BLOCK, 0.90), "injection"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
            (_make_decision(Action.ALLOW, 0.10), "benign"),
        ]
        report = EvalReport(predictions)
        cm = report.confusion_matrix()
        assert cm["TP"] == 2
        assert cm["TN"] == 2
        assert cm["FP"] == 0
        assert cm["FN"] == 0

    def test_all_false_positives(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "benign"),
            (_make_decision(Action.FLAG, 0.60), "benign"),
        ]
        report = EvalReport(predictions)
        cm = report.confusion_matrix()
        assert cm["FP"] == 2
        assert cm["TP"] == 0

    def test_all_false_negatives(self):
        predictions = [
            (_make_decision(Action.ALLOW, 0.10), "injection"),
            (_make_decision(Action.ALLOW, 0.20), "injection"),
        ]
        report = EvalReport(predictions)
        cm = report.confusion_matrix()
        assert cm["FN"] == 2
        assert cm["TP"] == 0


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 computation."""

    def test_perfect_metrics(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.BLOCK, 0.90), "injection"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
            (_make_decision(Action.ALLOW, 0.10), "benign"),
        ]
        report = EvalReport(predictions)
        metrics = report.precision_recall_f1()
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.fpr == 0.0
        assert metrics.fnr == 0.0

    def test_partial_metrics(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),  # TP
            (_make_decision(Action.ALLOW, 0.10), "injection"),  # FN
            (_make_decision(Action.BLOCK, 0.90), "benign"),     # FP
            (_make_decision(Action.ALLOW, 0.05), "benign"),     # TN
        ]
        report = EvalReport(predictions)
        metrics = report.precision_recall_f1()
        # TP=1, FP=1, FN=1, TN=1
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5
        assert metrics.f1 == 0.5
        assert metrics.fpr == 0.5
        assert metrics.fnr == 0.5

    def test_no_positives(self):
        predictions = [
            (_make_decision(Action.ALLOW, 0.05), "benign"),
            (_make_decision(Action.ALLOW, 0.10), "benign"),
        ]
        report = EvalReport(predictions)
        metrics = report.precision_recall_f1()
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0


class TestRocAuc:
    """Test ROC-AUC computation."""

    def test_perfect_separation(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.BLOCK, 0.90), "injection"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
            (_make_decision(Action.ALLOW, 0.10), "benign"),
        ]
        report = EvalReport(predictions)
        # Mock sklearn away to test manual AUC
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.metrics": None}):
            auc = report._manual_roc_auc()
        assert auc == 1.0

    def test_random_auc(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.5), "injection"),
            (_make_decision(Action.ALLOW, 0.5), "benign"),
        ]
        report = EvalReport(predictions)
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.metrics": None}):
            auc = report._manual_roc_auc()
        # Equal scores -> AUC = 0.5 (ties)
        assert auc == 0.5


class TestScoreDistribution:
    """Test score distribution histogram."""

    def test_basic_distribution(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
        ]
        report = EvalReport(predictions)
        dist = report.score_distribution(bins=10)
        assert len(dist["bin_edges"]) == 11
        assert sum(dist["injection_counts"]) == 1
        assert sum(dist["benign_counts"]) == 1


class TestRecommendThresholds:
    """Test threshold recommendation."""

    def test_recommends_thresholds(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.BLOCK, 0.85), "injection"),
            (_make_decision(Action.ALLOW, 0.15), "benign"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
        ]
        report = EvalReport(predictions)
        config = report.recommend_thresholds()
        assert 0.0 <= config.block_threshold <= 1.0
        assert 0.0 <= config.flag_threshold <= 1.0
        assert config.flag_threshold <= config.block_threshold

    def test_recommends_with_target_fpr(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.95), "injection"),
            (_make_decision(Action.ALLOW, 0.05), "benign"),
        ]
        report = EvalReport(predictions)
        config = report.recommend_thresholds(target_fpr=0.1)
        assert config.block_threshold > 0.0


class TestPerModelDiagnostics:
    """Test per-model metric computation."""

    def test_per_model_metrics(self):
        predictions = [
            (
                _make_decision(
                    Action.BLOCK, 0.95,
                    model_scores={"model-a": ClassifierResult(score=0.9, label="injection")},
                ),
                "injection",
            ),
            (
                _make_decision(
                    Action.ALLOW, 0.05,
                    model_scores={"model-a": ClassifierResult(score=0.1, label="benign")},
                ),
                "benign",
            ),
        ]
        report = EvalReport(predictions)
        diag = report.per_model_diagnostics()
        assert "model-a" in diag
        assert diag["model-a"].precision == 1.0
        assert diag["model-a"].recall == 1.0


class TestCalibrationCurves:
    """Test calibration curve computation."""

    def test_basic_calibration(self):
        predictions = [
            (_make_decision(Action.BLOCK, 0.9), "injection"),
            (_make_decision(Action.ALLOW, 0.1), "benign"),
        ]
        report = EvalReport(predictions)
        cal = report.calibration_curves(n_bins=10)
        assert len(cal["mean_predicted"]) == 10
        assert len(cal["fraction_positive"]) == 10
        assert len(cal["bin_counts"]) == 10
