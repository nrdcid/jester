"""
Impurity metrics for decision tree splitting.
"""
import numpy as np


def gini(y):
    """
    Calculate the Gini impurity of labels.

    Gini impurity measures the probability of incorrectly classifying a randomly
    chosen element if it was randomly labeled according to the distribution of
    labels in the subset.

    Formula: Gini(y) = 1 - sum(p_c^2) for all classes c
    where p_c is the probability of class c

    Args:
        y: 1-d array containing labels

    Returns:
        float: Gini impurity value between 0 (pure) and ~0.5 (maximum impurity)
    """
    return 1 - sum((np.bincount(y) / len(y))**2)


def impurity_reduction(y, left_indices, right_indices):
    """
    Calculate the reduction in impurity from a split.

    Computes the weighted reduction in Gini impurity achieved by splitting
    the dataset into left and right subsets.

    Formula:
        reduction = gini(parent) - [|left|/|parent| * gini(left) + |right|/|parent| * gini(right)]

    Args:
        y: 1-d array containing all labels
        left_indices: indices of samples going to the left child
        right_indices: indices of samples going to the right child

    Returns:
        float: The impurity reduction value (positive means improvement)
    """
    left_impurity = gini(y[left_indices])
    right_impurity = gini(y[right_indices])
    total_impurity = gini(y)

    weighted_child_impurity = (
        (len(left_indices) / len(y)) * left_impurity +
        (len(right_indices) / len(y)) * right_impurity
    )

    return total_impurity - weighted_child_impurity
