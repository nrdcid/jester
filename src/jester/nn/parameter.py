"""
Parameter class for neural network trainable parameters.

This module provides a Parameter class that extends numpy arrays with gradient tracking
capabilities, enabling automatic differentiation for neural network training.
"""

import numpy as np


class Parameter(np.ndarray):
    """
    A numpy array subclass that tracks gradients for neural network parameters.

    This class wraps numpy arrays and adds:
    - gradient: An array of the same shape to store gradients
    - name: An optional identifier for the parameter
    - Methods to manage gradients (zero, apply)

    Attributes:
        gradient (np.ndarray): Gradient array with same shape as parameter
        name (str): Optional name identifier for the parameter
    """

    def __new__(cls, input_array, name=""):
        """
        Create a new Parameter instance.

        Args:
            input_array: Array-like object to convert to Parameter
            name: Optional name for the parameter

        Returns:
            Parameter instance
        """
        array = np.asarray(input_array).view(cls)
        array.gradient = np.zeros(array.shape)
        array.name = name
        return array

    def __array_finalize__(self, array):
        """
        Finalize array creation, ensuring gradient and name attributes exist.

        Args:
            array: Array being finalized
        """
        if array is None:
            return
        self.gradient = getattr(array, "gradient", None)
        self.name = getattr(array, "name", None)

    def zero_gradient(self):
        """Reset gradient to zeros."""
        self.gradient = self.gradient * 0.0

    def apply_gradient(self, grad):
        """
        Apply gradient update to parameter values.

        Args:
            grad: Gradient array to add to current parameter values
        """
        self[:] = self[:] + grad[:]
