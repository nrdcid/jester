"""
Decision Trees - A production-ready decision tree classifier implementation.

This package provides a complete implementation of decision tree classification
with support for Gini impurity, recursive tree building, and feature importance.
"""

from .tree import DecisionTree, build_tree
from .nodes import LeafNode, DecisionNode
from .metrics import gini, impurity_reduction
from .splitting import best_split
from .utils import compute_label, split_values

__version__ = "1.0.0"
__all__ = [
    "DecisionTree",
    "build_tree",
    "LeafNode",
    "DecisionNode",
    "gini",
    "impurity_reduction",
    "best_split",
    "compute_label",
    "split_values",
]
