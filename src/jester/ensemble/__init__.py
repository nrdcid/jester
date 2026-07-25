"""
Ensemble Methods - Production-ready ensemble learning implementations.

This package provides implementations of popular ensemble methods including
Bagging, Random Forest, and AdaBoost, along with utilities for evaluation.
"""

from .bagging import BaggingEnsemble
from .random_forest import RandomForest
from .adaboost import AdaBoost
from .evaluator import Evaluator
from .utils import get_weak_learner, random_selection

__version__ = "1.0.0"
__all__ = [
    "BaggingEnsemble",
    "RandomForest",
    "AdaBoost",
    "Evaluator",
    "get_weak_learner",
    "random_selection",
]
