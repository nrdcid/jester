"""Impurity metrics for decision tree splitting."""
import numpy as np


def _class_probabilities(y):
    """Return class frequencies for labels of any dtype."""
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return np.empty(0, dtype=float)
    _, counts = np.unique(y, return_counts=True)
    return counts / y.size


def gini(y):
    """Return Gini impurity, supporting negative and string labels."""
    probabilities = _class_probabilities(y)
    if probabilities.size == 0:
        return 0.0
    return float(1.0 - np.sum(probabilities ** 2))


def entropy(y):
    """Return Shannon entropy in bits."""
    probabilities = _class_probabilities(y)
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


def misclassification(y):
    """Return node misclassification error."""
    probabilities = _class_probabilities(y)
    if probabilities.size == 0:
        return 0.0
    return float(1.0 - probabilities.max())


CRITERIA = {
    "gini": gini,
    "entropy": entropy,
    "misclassification": misclassification,
}


def resolve_criterion(criterion):
    """Resolve a criterion name or callable to an impurity function."""
    if callable(criterion):
        return criterion
    try:
        return CRITERIA[criterion]
    except KeyError:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected a callable or one of "
            f"{sorted(CRITERIA)}"
        ) from None


def impurity_reduction(y, left_indices, right_indices, criterion=gini):
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
    impurity = resolve_criterion(criterion)
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return 0.0

    left_impurity = impurity(y[left_indices])
    right_impurity = impurity(y[right_indices])
    total_impurity = impurity(y)

    weighted_child_impurity = (
        (len(left_indices) / len(y)) * left_impurity +
        (len(right_indices) / len(y)) * right_impurity
    )

    return float(total_impurity - weighted_child_impurity)
