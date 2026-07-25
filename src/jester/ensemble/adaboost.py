"""
AdaBoost (Adaptive Boosting) ensemble method implementation.
"""
import numpy as np
from .bagging import BaggingEnsemble
from .utils import get_weak_learner


class AdaBoost(BaggingEnsemble):
    """
    AdaBoost (SAMME variant) classifier.

    AdaBoost sequentially trains weak learners, with each new learner
    focusing more on samples that were misclassified by previous learners.
    Sample weights are adaptively updated based on prediction errors.
    """

    def __init__(self, n_estimators):
        """
        Initialize the AdaBoost ensemble.

        Args:
            n_estimators: Number of boosting rounds
        """
        super(AdaBoost, self).__init__(n_estimators)
        self.num_classes = None
        self.alphas = []  # Estimator weights
        self.classes_ = None

    def fit(self, X_train, y_train):
        """
        Train the AdaBoost ensemble using the SAMME algorithm.

        SAMME (Stagewise Additive Modeling using a Multi-class Exponential
        loss function) extends AdaBoost to multi-class problems.

        Algorithm:
        1. Initialize equal sample weights
        2. For each boosting round:
           - Train weak learner with current weights
           - Calculate weighted error rate
           - Compute estimator weight alpha
           - Update sample weights (increase for misclassified samples)
           - Normalize weights

        Args:
            X_train: Training features of shape (num_samples, num_features)
            y_train: Training labels of shape (num_samples,)

        Returns:
            self: The fitted ensemble
        """
        self.estimators = []
        self.alphas = []
        self.classes_ = np.unique(y_train)
        self.num_classes = self.classes_.shape[0]

        # Initialize equal weights for all samples
        weights = np.ones(len(X_train)) / len(X_train)

        for _ in range(self.n_estimators):
            # Train weak learner with current sample weights
            model = get_weak_learner()
            model.fit(X_train, y_train, sample_weight=weights)

            # Get predictions and calculate weighted error
            y_pred = model.predict(X_train)
            error = np.sum(weights * (y_pred != y_train)) / np.sum(weights)

            # Clip error to avoid division by zero or log(0)
            error = np.clip(error, 1e-12, 1 - 1e-12)

            # Calculate estimator weight (SAMME formula)
            alpha = np.log((1 - error) / error) + np.log(self.num_classes - 1)

            # Update sample weights (exponentially increase for misclassified)
            weights *= np.exp(alpha * (y_pred != y_train))

            # Normalize weights to sum to 1
            weights /= np.sum(weights)

            self.estimators.append(model)
            self.alphas.append(alpha)

        return self

    def predict(self, X_test):
        """
        Predict labels using weighted voting.

        Each estimator's prediction is weighted by its alpha value.
        The class with the highest weighted vote is returned.

        Args:
            X_test: Test features of shape (num_samples, num_features)

        Returns:
            ndarray: Predicted labels of shape (num_samples,)
        """
        scores = np.zeros((X_test.shape[0], self.num_classes))

        for model, alpha in zip(self.estimators, self.alphas):
            pred = model.predict(X_test).astype(int)

            # Add weighted votes for predicted classes
            for idx, cls in enumerate(self.classes_):
                mask = pred == cls
                if np.any(mask):
                    scores[mask, idx] += alpha

        answer = self.classes_[np.argmax(scores, axis=1)]

        return answer
