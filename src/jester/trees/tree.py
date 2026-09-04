"""
Decision tree classifier implementation.
"""
import numpy as np
from .nodes import LeafNode, RegressionLeafNode, DecisionNode
from .splitting import best_split


def build_tree(X, y, depth=-1, min_samples_split=2, criterion="gini"):
    """
    Recursively build a decision tree.

    Args:
        X: 2-d array of shape (num_samples, num_features)
        y: 1-d array of labels
        depth: Maximum depth to build (-1 for unlimited)
        min_samples_split: Minimum samples required to split a node

    Returns:
        Node: Root node of the constructed tree (LeafNode or DecisionNode)
    """
    if depth == 0 or len(y) < min_samples_split:
        # Base case: maximum depth reached or too few samples
        return LeafNode(y)

    feature_id, threshold, left_indices, right_indices, reduction = best_split(
        X, y, criterion=criterion
    )

    if reduction <= 0:
        return LeafNode(y)

    left_child = build_tree(
        X[left_indices], y[left_indices], depth - 1, min_samples_split, criterion
    )
    right_child = build_tree(
        X[right_indices], y[right_indices], depth - 1, min_samples_split, criterion
    )
    return DecisionNode(feature_id, threshold, left_child, right_child)


def build_tree_regressor(
    X, y, depth=-1, min_samples_split=2, criterion="squared_error"
):
    """Recursively build a regression tree using variance reduction."""
    if depth == 0 or len(y) < min_samples_split:
        return RegressionLeafNode(y)

    feature_id, threshold, left_indices, right_indices, reduction = best_split(
        X, y, criterion=criterion
    )
    if reduction <= 0:
        return RegressionLeafNode(y)

    left_child = build_tree_regressor(
        X[left_indices], y[left_indices], depth - 1, min_samples_split, criterion
    )
    right_child = build_tree_regressor(
        X[right_indices], y[right_indices], depth - 1, min_samples_split, criterion
    )
    return DecisionNode(feature_id, threshold, left_child, right_child)


class DecisionTree:
    """
    Decision tree classifier.

    A decision tree recursively splits the feature space to separate
    different classes, making predictions based on the majority class
    in each leaf region.
    """

    def __init__(self, max_depth=-1, min_samples_split=2, criterion="gini"):
        """
        Initialize the decision tree classifier.

        Args:
            max_depth: Maximum depth of the tree (-1 for unlimited)
            min_samples_split: Minimum number of samples required for a split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.num_features = None

    def fit(self, X, y):
        """
        Build the decision tree from training data.

        Args:
            X: Training samples of shape (num_samples, num_features)
            y: Training labels of shape (num_samples,)

        Returns:
            self: The fitted decision tree
        """
        self.tree = build_tree(
            X, y, self.max_depth, self.min_samples_split, self.criterion
        )
        return self

    def predict(self, X):
        """
        Predict labels for given samples.

        Args:
            X: Samples to predict, shape (num_samples, num_features)

        Returns:
            ndarray: Predicted labels of shape (num_samples,)
        """
        return self.tree.predict(X)

    def feature_importance(self, X, y):
        """
        Compute the importance of each feature in the decision tree.

        Feature importance is measured by the total weighted reduction
        in impurity contributed by each feature across all splits.

        Args:
            X: Feature matrix of shape (num_samples, num_features)
            y: Labels of shape (num_samples,)

        Returns:
            dict: Dictionary mapping feature index to normalized importance
        """
        feat_importance = {k: 0 for k in range(X.shape[1])}
        if not self.tree.is_terminal():
            self.tree.add_importance(feat_importance, X, y, self.criterion)
        # Normalize to sum to 1
        total = sum(feat_importance.values())
        if total > 0:
            feat_importance = {k: v / total for k, v in feat_importance.items()}
        return feat_importance

    def score(self, X, y):
        """
        Calculate accuracy on given data.

        Args:
            X: Samples of shape (num_samples, num_features)
            y: True labels of shape (num_samples,)

        Returns:
            float: Accuracy score (fraction of correct predictions)
        """
        predicted_labels = self.predict(X)
        accuracy = np.mean(predicted_labels == y)
        return accuracy


class DecisionTreeRegressor:
    """Decision tree regressor with mean-valued leaves."""

    def __init__(self, max_depth=-1, min_samples_split=2, criterion="squared_error"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and have the same number of rows as y")
        self.tree = build_tree_regressor(
            X, y, self.max_depth, self.min_samples_split, self.criterion
        )
        self.num_features = X.shape[1]
        return self

    def predict(self, X):
        if self.tree is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.tree.predict(np.asarray(X)), dtype=float)

    def feature_importance(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        importances = {k: 0.0 for k in range(X.shape[1])}
        if not self.tree.is_terminal():
            self.tree.add_importance(importances, X, y, self.criterion)
        total = sum(importances.values())
        if total > 0:
            importances = {k: value / total for k, value in importances.items()}
        return importances

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        predictions = self.predict(X)
        total = np.sum((y - np.mean(y)) ** 2)
        if total == 0:
            return 1.0 if np.allclose(predictions, y) else 0.0
        return float(1.0 - np.sum((y - predictions) ** 2) / total)
