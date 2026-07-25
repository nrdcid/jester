"""Shared evaluation metrics."""

from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    """Fraction of matching labels."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """
    Build a square confusion matrix.

    Rows are true labels, columns are predicted labels.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix
