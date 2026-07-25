"""
Utility functions for decision tree construction.
"""
import numpy as np


def compute_label(node_labels):
    """
    Compute the label for a leaf node.

    The label is determined by the most frequent class in node_labels.
    If there's a tie, the class with the highest value is chosen.

    Args:
        node_labels: 1-d array containing labels at this node

    Returns:
        int: The label to assign to this leaf node
    """
    unique_labels, counts = np.unique(node_labels, return_counts=True)
    max_count = np.max(counts)
    most_frequent_labels = unique_labels[counts == max_count]
    label = np.max(most_frequent_labels)

    return label


def split_values(feature_values):
    """
    Generate candidate split thresholds for a feature.

    Returns midpoints between consecutive unique values in the feature.
    This is used for numeric features in decision tree splitting.

    Args:
        feature_values: 1-d array of feature values

    Returns:
        ndarray: Array of candidate split thresholds
    """
    unique_values = np.unique(feature_values)
    if unique_values.shape[0] == 1:
        return unique_values
    return (unique_values[1:] + unique_values[:-1]) / 2
