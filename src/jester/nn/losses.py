"""
Loss function implementations for neural networks.

This module provides loss functions used to train neural networks,
including forward pass (loss computation) and backward pass (gradient computation).
"""

import numpy as np


class MSE:
    """
    Mean Squared Error (MSE) loss function.

    Computes the mean squared error between predictions and targets:
    MSE = (1/m) * sum((y_pred - y_true)^2)

    The gradient with respect to predictions is:
    dMSE/dy_pred = (2/m) * (y_pred - y_true)

    Attributes:
        saved_arrays (list): Caches y_pred and y_true for backward pass
    """

    def __init__(self):
        """Initialize MSE loss function."""
        self.saved_arrays = []

    def forward(self, y_pred, y_true):
        """
        Compute the mean squared error loss.

        Args:
            y_pred: Predicted values, shape (batch_size, output_dim)
            y_true: True target values, shape (batch_size, output_dim)

        Returns:
            Scalar loss value (mean squared error)
        """
        self.saved_arrays = [y_pred, y_true]
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        """
        Compute gradient of MSE loss with respect to predictions.

        Uses cached values from forward pass to compute:
        dL/dy_pred = (2/m) * (y_pred - y_true)

        Returns:
            Gradient array of same shape as y_pred
        """
        y_pred, y_true = self.saved_arrays
        self.saved_arrays = []
        return (2 * (y_pred - y_true)) / len(y_pred)
