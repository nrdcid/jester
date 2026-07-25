"""
Utility functions for ensemble methods.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_weak_learner():
    """
    Return a new instance of the chosen weak learner.

    Uses a shallow decision tree with max_depth=3 and min_samples_leaf=0.1
    as a weak learner for ensemble methods.

    Returns:
        DecisionTreeClassifier: A configured weak learner instance
    """
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)


def random_selection(input_dim, output_dim):
    """
    Randomly select features for Random Forest.

    Selects output_dim features from input_dim features with replacement.
    Used in Random Forest to create random feature subsets for each estimator.

    Args:
        input_dim: Number of input features
        output_dim: Number of features to select

    Returns:
        ndarray: Array of selected feature indices
    """
    assert input_dim >= output_dim, "Output dimension must be <= input dimension"
    return np.random.choice(input_dim, size=output_dim, replace=True)
