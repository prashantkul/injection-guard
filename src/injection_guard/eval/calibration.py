"""Calibration utilities for post-hoc probability calibration."""
from __future__ import annotations

from injection_guard.types import EvalMetrics

__all__ = ["PlattScaler", "IsotonicCalibrator"]


class PlattScaler:
    """Platt scaling (logistic regression) for probability calibration.

    Fits a sigmoid to map raw classifier scores to calibrated probabilities.
    Uses scikit-learn's LogisticRegression when available, otherwise falls
    back to a simple sigmoid fit via Newton's method.
    """

    def __init__(self) -> None:
        self._a: float = 0.0
        self._b: float = 0.0
        self._fitted: bool = False
        self._lr: object | None = None

    def fit(self, scores: list[float], labels: list[int]) -> PlattScaler:
        """Fit the Platt scaler on raw scores and binary labels.

        Args:
            scores: Raw classifier scores in [0, 1].
            labels: Ground-truth binary labels (0 or 1).

        Returns:
            self for chaining.
        """
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length")
        if len(scores) == 0:
            raise ValueError("Cannot fit on empty data")

        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            X = np.array(scores).reshape(-1, 1)
            y = np.array(labels)
            lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
            lr.fit(X, y)
            self._lr = lr
            self._fitted = True
        except ImportError:
            self._fit_manual(scores, labels)

        return self

    def _fit_manual(self, scores: list[float], labels: list[int]) -> None:
        """Manual Platt scaling via gradient descent on log-loss."""
        import math

        a = 0.0
        b = 0.0
        lr = 0.01
        n = len(scores)

        # Target probabilities with Bayesian smoothing
        pos = sum(labels)
        neg = n - pos
        hi_target = (pos + 1) / (pos + 2)
        lo_target = 1 / (neg + 2)
        targets = [hi_target if lab == 1 else lo_target for lab in labels]

        for _ in range(5000):
            grad_a = 0.0
            grad_b = 0.0
            for s, t in zip(scores, targets):
                fval = a * s + b
                fval = max(-30.0, min(30.0, fval))
                p = 1.0 / (1.0 + math.exp(-fval))
                grad_a += (p - t) * s
                grad_b += (p - t)
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self._a = a
        self._b = b
        self._fitted = True

    def transform(self, scores: list[float]) -> list[float]:
        """Transform raw scores to calibrated probabilities.

        Args:
            scores: Raw classifier scores.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("PlattScaler has not been fitted yet")

        if self._lr is not None:
            try:
                import numpy as np

                X = np.array(scores).reshape(-1, 1)
                return self._lr.predict_proba(X)[:, 1].tolist()
            except Exception:
                pass

        import math

        result: list[float] = []
        for s in scores:
            fval = self._a * s + self._b
            fval = max(-30.0, min(30.0, fval))
            p = 1.0 / (1.0 + math.exp(-fval))
            result.append(p)
        return result


class IsotonicCalibrator:
    """Isotonic regression for probability calibration.

    Uses scikit-learn's IsotonicRegression when available, otherwise falls
    back to a simple pool-adjacent-violators (PAVA) implementation.
    """

    def __init__(self) -> None:
        self._ir: object | None = None
        self._fitted: bool = False
        self._x: list[float] = []
        self._y: list[float] = []

    def fit(self, scores: list[float], labels: list[int]) -> IsotonicCalibrator:
        """Fit the isotonic calibrator on raw scores and binary labels.

        Args:
            scores: Raw classifier scores in [0, 1].
            labels: Ground-truth binary labels (0 or 1).

        Returns:
            self for chaining.
        """
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length")
        if len(scores) == 0:
            raise ValueError("Cannot fit on empty data")

        try:
            from sklearn.isotonic import IsotonicRegression
            import numpy as np

            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(np.array(scores), np.array(labels, dtype=float))
            self._ir = ir
            self._fitted = True
        except ImportError:
            self._fit_manual(scores, labels)

        return self

    def _fit_manual(self, scores: list[float], labels: list[int]) -> None:
        """Pool-adjacent-violators algorithm (PAVA) implementation."""
        # Sort by score
        pairs = sorted(zip(scores, labels), key=lambda p: p[0])
        x_sorted = [p[0] for p in pairs]
        y_sorted = [float(p[1]) for p in pairs]

        # PAVA
        n = len(y_sorted)
        result = list(y_sorted)
        weight = [1.0] * n

        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Pool blocks
                merged_val = (result[i] * weight[i] + result[i + 1] * weight[i + 1]) / (
                    weight[i] + weight[i + 1]
                )
                merged_w = weight[i] + weight[i + 1]
                result[i] = merged_val
                weight[i] = merged_w
                result.pop(i + 1)
                weight.pop(i + 1)
                x_sorted.pop(i + 1)
                n -= 1
                # Step back to check previous
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Clamp to [0, 1]
        result = [max(0.0, min(1.0, v)) for v in result]

        self._x = x_sorted
        self._y = result
        self._fitted = True

    def transform(self, scores: list[float]) -> list[float]:
        """Transform raw scores to calibrated probabilities.

        Args:
            scores: Raw classifier scores.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted yet")

        if self._ir is not None:
            try:
                import numpy as np

                return self._ir.transform(np.array(scores)).tolist()
            except Exception:
                pass

        # Manual interpolation using fitted piecewise-constant function
        result: list[float] = []
        for s in scores:
            if not self._x:
                result.append(s)
                continue
            if s <= self._x[0]:
                result.append(self._y[0])
            elif s >= self._x[-1]:
                result.append(self._y[-1])
            else:
                # Linear interpolation between nearest points
                for j in range(len(self._x) - 1):
                    if self._x[j] <= s <= self._x[j + 1]:
                        if self._x[j + 1] == self._x[j]:
                            result.append(self._y[j])
                        else:
                            t = (s - self._x[j]) / (self._x[j + 1] - self._x[j])
                            result.append(self._y[j] + t * (self._y[j + 1] - self._y[j]))
                        break
                else:
                    result.append(self._y[-1])

        return result
