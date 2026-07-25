"""
Node classes for decision tree structure.
"""
import numpy as np
from .utils import compute_label
from .metrics import impurity_reduction


class LeafNode:
    """
    Terminal node in a decision tree.

    Leaf nodes hold a single label value that is predicted for all
    samples reaching this node.
    """

    def __init__(self, node_labels):
        """
        Initialize the leaf node.

        Args:
            node_labels: 1-d array containing labels of samples at this node
        """
        self.label = compute_label(node_labels)

    @staticmethod
    def is_terminal():
        """Check if this node is a terminal (leaf) node."""
        return True

    def predict(self, X):
        """
        Predict labels for given samples.

        Args:
            X: 2-d array of shape (num_samples, num_features)

        Returns:
            ndarray: Array of predicted labels (all equal to self.label)
        """
        return self.label * np.ones(X.shape[0])


class DecisionNode:
    """
    Internal (non-terminal) node in a decision tree.

    Decision nodes split samples based on a feature threshold,
    routing them to left or right child nodes.
    """

    def __init__(self, feature_id, threshold, left_child, right_child):
        """
        Initialize the decision node.

        Args:
            feature_id: Index of the feature to split on
            threshold: Threshold value for the split
            left_child: Child node for samples with feature <= threshold
            right_child: Child node for samples with feature > threshold
        """
        self.feature_id = feature_id
        self.threshold = threshold
        self.left = left_child
        self.right = right_child

    @staticmethod
    def is_terminal():
        """Check if this node is a terminal (leaf) node."""
        return False

    def add_importance(self, importances, X, y):
        """
        Recursively compute feature importances.

        Feature importance is the weighted reduction in impurity
        contributed by each feature across all splits.

        Args:
            importances: Dictionary mapping feature_id to importance value
            X: Feature matrix for samples at this node
            y: Labels for samples at this node

        Returns:
            dict: Updated importances dictionary
        """
        left_indices = np.where(X[:, self.feature_id] <= self.threshold)[0]
        right_indices = np.where(X[:, self.feature_id] > self.threshold)[0]
        reduction = impurity_reduction(y, left_indices, right_indices)
        importances[self.feature_id] += len(y) * reduction

        if not self.left.is_terminal():
            self.left.add_importance(importances, X[left_indices], y[left_indices])
        if not self.right.is_terminal():
            self.right.add_importance(importances, X[right_indices], y[right_indices])

        return importances

    def predict(self, X):
        """
        Predict labels for given samples.

        Routes each sample to the appropriate child based on the split
        threshold, then recursively predicts from the child nodes.

        Args:
            X: 2-d array of shape (num_samples, num_features)

        Returns:
            ndarray: Array of predicted labels
        """
        y_pred = np.zeros((X.shape[0]))
        left_indices = np.where(X[:, self.feature_id] <= self.threshold)[0]
        right_indices = np.where(X[:, self.feature_id] > self.threshold)[0]
        y_pred[left_indices] = self.left.predict(X[left_indices])
        y_pred[right_indices] = self.right.predict(X[right_indices])
        return y_pred
