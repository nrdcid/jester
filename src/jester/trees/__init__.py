"""
Decision Trees - A production-ready decision tree classifier implementation.

This package provides a complete implementation of decision tree classification
with support for Gini impurity, recursive tree building, and feature importance.
"""

from .tree import DecisionTree, DecisionTreeRegressor, build_tree, build_tree_regressor
from .nodes import LeafNode, RegressionLeafNode, DecisionNode
from .metrics import (
    CRITERIA,
    REGRESSION_CRITERIA,
    entropy,
    gini,
    impurity_reduction,
    misclassification,
    resolve_criterion,
    variance,
)
from .splitting import best_split
from .utils import compute_label, split_values

__version__ = "1.0.0"
__all__ = [
    "DecisionTree",
    "DecisionTreeRegressor",
    "build_tree",
    "build_tree_regressor",
    "LeafNode",
    "RegressionLeafNode",
    "DecisionNode",
    "gini",
    "entropy",
    "misclassification",
    "variance",
    "CRITERIA",
    "REGRESSION_CRITERIA",
    "resolve_criterion",
    "impurity_reduction",
    "best_split",
    "compute_label",
    "split_values",
]
