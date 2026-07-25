"""
Functions for finding optimal splits in decision trees.
"""
import numpy as np
from .utils import split_values
from .metrics import impurity_reduction


def best_split(X, y):
    """
    Find the best feature and threshold for splitting the data.

    Searches through all features and candidate thresholds to find
    the split that maximizes impurity reduction.

    Args:
        X: 2-d array of shape (num_samples, num_features)
        y: 1-d array of labels

    Returns:
        tuple: (best_feature_id, best_threshold, best_left_indices,
                best_right_indices, best_reduction)
               - best_feature_id: Index of the best feature to split on
               - best_threshold: Optimal threshold value
               - best_left_indices: Indices of samples going left
               - best_right_indices: Indices of samples going right
               - best_reduction: Impurity reduction achieved by this split
    """
    best_feature_id, best_threshold = None, None
    best_left_indices, best_right_indices = None, None
    best_reduction = -np.inf

    for feature_id in range(X.shape[1]):
        for threshold in split_values(X[:, feature_id]):
            left_indices = np.where(X[:, feature_id] <= threshold)[0]
            right_indices = np.where(X[:, feature_id] > threshold)[0]
            reduction = impurity_reduction(y, left_indices, right_indices)

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature_id = feature_id
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices

    return best_feature_id, best_threshold, best_left_indices, best_right_indices, best_reduction
