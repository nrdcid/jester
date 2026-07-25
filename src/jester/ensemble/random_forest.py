"""
Random Forest ensemble method implementation.
"""
import numpy as np
from .bagging import BaggingEnsemble
from .utils import get_weak_learner, random_selection


class RandomForest(BaggingEnsemble):
    """
    Random Forest classifier.

    Extends Bagging by also randomly selecting a subset of features for each
    estimator, creating additional diversity in the ensemble.
    """

    def __init__(self, n_estimators, sample_ratio=1.0, features_ratio=1.0):
        """
        Initialize the random forest.

        Args:
            n_estimators: Number of trees in the forest
            sample_ratio: Ratio of training samples to use per tree
            features_ratio: Ratio of features to use per tree
        """
        super(RandomForest, self).__init__(n_estimators, sample_ratio)
        self.features_ratio = features_ratio
        self.estimators = []
        self.selections = []  # Stores feature indices for each estimator
        self.classes_ = None

    def sample_data(self, X_train, y_train):
        """
        Create a bootstrap sample with random feature selection.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            tuple: (X_sample, y_sample, selected_features)
                   - X_sample: Sampled and projected features
                   - y_sample: Sampled labels
                   - selected_features: Indices of selected features
        """
        input_dim = X_train.shape[1]
        output_dim = max(1, int(np.ceil(self.features_ratio * input_dim)))
        indices = np.random.choice(
            len(X_train),
            int(self.sample_ratio * len(X_train)),
            replace=True
        )
        selected_features = random_selection(input_dim, output_dim)

        return X_train[indices][:, selected_features], y_train[indices], selected_features

    def fit(self, X_train, y_train):
        """
        Train the random forest.

        Creates n_estimators weak learners, each trained on a bootstrap sample
        with a random subset of features.

        Args:
            X_train: Training features of shape (num_samples, num_features)
            y_train: Training labels of shape (num_samples,)

        Returns:
            self: The fitted forest
        """
        self.estimators = []
        self.selections = []
        self.classes_ = np.unique(y_train)

        for _ in range(self.n_estimators):
            X_sub, y_sub, selected = self.sample_data(X_train, y_train)
            model = get_weak_learner()
            model.fit(X_sub, y_sub)
            self.estimators.append(model)
            self.selections.append(selected)

        return self

    def predict(self, X_test):
        """
        Predict labels by aggregating forest predictions.

        Each tree operates on its selected feature subset. Predictions are
        aggregated through soft voting across all trees.

        Args:
            X_test: Test features of shape (num_samples, num_features)

        Returns:
            ndarray: Predicted labels of shape (num_samples,)
        """
        predicted_proba = None

        for model, selected in zip(self.estimators, self.selections):
            proba = model.predict_proba(X_test[:, selected])

            if predicted_proba is None:
                predicted_proba = np.zeros((X_test.shape[0], len(self.classes_)))

            # Map estimator's classes to global class indices
            for est_col, cls in enumerate(model.classes_):
                global_col = np.where(self.classes_ == cls)[0][0]
                predicted_proba[:, global_col] += proba[:, est_col]

        answer = self.classes_[np.argmax(predicted_proba, axis=1)]

        return answer
