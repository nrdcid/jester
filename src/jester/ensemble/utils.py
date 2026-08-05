"""
Utility functions for ensemble methods.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_weak_learner(factory=None):
    """
    Return a new instance of the chosen weak learner.

    Uses a shallow decision tree with max_depth=3 and min_samples_leaf=0.1
    as a weak learner for ensemble methods.

    Returns:
        DecisionTreeClassifier: A configured weak learner instance
    """
    if factory is not None:
        return factory()
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)


def random_selection(input_dim, output_dim, rng=None):
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
    chooser = np.random.choice if rng is None else rng.choice
    return chooser(input_dim, size=output_dim, replace=True)
