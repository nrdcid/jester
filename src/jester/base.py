"""sklearn-style estimator interface for future API unification.

Models are not required to subclass this yet. Prefer these method names
when adding or revising estimators: ``fit``, ``predict``, ``score``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EstimatorMixin(ABC):
    """Minimal estimator contract: fit / predict / score."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model. Should return ``self``."""

    @abstractmethod
    def predict(self, X):
        """Return predictions for ``X``."""

    def score(self, X, y):
        """
        Default score: mean accuracy for discrete labels, else negative RMSE.

        Override when the task needs a different metric.
        """
        from jester.metrics import accuracy, rmse

        y_pred = np.asarray(self.predict(X)).reshape(-1)
        y_true = np.asarray(y).reshape(-1)

        if y_true.dtype.kind in "iub" or np.array_equal(y_true, y_true.astype(int)):
            unique = np.unique(y_true)
            if unique.size <= max(50, int(0.05 * y_true.size) + 1):
                return accuracy(y_true, y_pred)

        return -rmse(y_true, y_pred)
