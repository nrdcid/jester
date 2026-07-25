"""Jester: machine learning algorithms implemented from scratch."""

from jester.base import EstimatorMixin
from jester.metrics import accuracy, confusion_matrix, mse, rmse
from jester.viz import display_confusion, plot_data, show_decision_surface

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "EstimatorMixin",
    "accuracy",
    "mse",
    "rmse",
    "confusion_matrix",
    "show_decision_surface",
    "display_confusion",
    "plot_data",
]
