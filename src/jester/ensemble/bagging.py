"""
Bagging (Bootstrap Aggregating) ensemble method implementation.
"""
import numpy as np
from .utils import get_weak_learner


class BaggingEnsemble:
    """
    Bagging ensemble classifier.

    Bagging trains multiple estimators on random subsets of the training data
    (sampled with replacement) and aggregates their predictions through voting.
    """

    def __init__(self, n_estimators, sample_ratio=1.0):
        """
        Initialize the bagging ensemble.

        Args:
            n_estimators: Number of base estimators to train
            sample_ratio: Ratio of training data to sample for each estimator
        """
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.estimators = []

    def sample_data(self, X_train, y_train):
        """
        Create a bootstrap sample of the training data.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            tuple: (X_sample, y_sample) - Sampled features and labels
        """
        size = int(self.sample_ratio * len(X_train))
        indices = np.random.choice(len(X_train), size, replace=True)

        X_sample = X_train[indices]
        y_sample = y_train[indices]

        return X_sample, y_sample

    def fit(self, X_train, y_train):
        """
        Train the bagging ensemble.

        Creates n_estimators weak learners, each trained on a bootstrap
        sample of the training data.

        Args:
            X_train: Training features of shape (num_samples, num_features)
            y_train: Training labels of shape (num_samples,)

        Returns:
            self: The fitted ensemble
        """
        np.random.seed(42)  # For reproducibility
        self.estimators = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self.sample_data(X_train, y_train)
            model = get_weak_learner()
            model.fit(X_sample, y_sample)
            self.estimators.append(model)

        return self

    def predict(self, X_test):
        """
        Predict labels by aggregating estimator predictions.

        Uses soft voting: sums prediction probabilities from all estimators
        and returns the class with highest total probability.

        Args:
            X_test: Test features of shape (num_samples, num_features)

        Returns:
            ndarray: Predicted labels of shape (num_samples,)
        """
        predicted_proba = 0

        for model in self.estimators:
            predicted_proba += model.predict_proba(X_test)

        answer = np.argmax(predicted_proba, axis=1)

        return answer

    def score(self, X, y):
        """
        Calculate accuracy on given data.

        Args:
            X: Features of shape (num_samples, num_features)
            y: True labels of shape (num_samples,)

        Returns:
            float: Accuracy score
        """
        return np.mean(self.predict(X) == y)
