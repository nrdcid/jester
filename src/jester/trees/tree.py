"""
Decision tree classifier implementation.
"""
import numpy as np
from .nodes import LeafNode, DecisionNode
from .splitting import best_split


def build_tree(X, y, depth=-1, min_samples_split=2):
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
        tree = LeafNode(y)
    else:
        feature_id, threshold, left_indices, right_indices, reduction = best_split(X, y)

        if reduction <= 0:
            # No beneficial split found
            tree = LeafNode(y)
        else:
            # Recursively build left and right subtrees
            left_child = build_tree(
                X[left_indices], y[left_indices],
                depth - 1, min_samples_split
            )
            right_child = build_tree(
                X[right_indices], y[right_indices],
                depth - 1, min_samples_split
            )
            tree = DecisionNode(feature_id, threshold, left_child, right_child)

    return tree


class DecisionTree:
    """
    Decision tree classifier.

    A decision tree recursively splits the feature space to separate
    different classes, making predictions based on the majority class
    in each leaf region.
    """

    def __init__(self, max_depth=-1, min_samples_split=2):
        """
        Initialize the decision tree classifier.

        Args:
            max_depth: Maximum depth of the tree (-1 for unlimited)
            min_samples_split: Minimum number of samples required for a split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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
        self.tree = build_tree(X, y, self.max_depth, self.min_samples_split)
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
            self.tree.add_importance(feat_importance, X, y)
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
