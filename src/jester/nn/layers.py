"""
Neural network layer implementations.

This module provides the base Layer class and specific layer implementations
including Dense (fully connected), Sigmoid, and ReLU activation layers.
"""

import numpy as np
from .parameter import Parameter


class Layer(object):
    """
    Base class for all neural network layers.

    Provides the interface that all layers must implement with forward
    and backward pass methods.

    Attributes:
        saved_arrays (list): Cache for storing values needed in backward pass
        parameters (list): List of trainable Parameter objects
        name (str): Identifier for the layer
    """

    def __init__(self, name=""):
        """
        Initialize base layer.

        Args:
            name: Optional identifier for the layer
        """
        self.saved_arrays = []  # Cache for backward pass
        self.parameters = []    # Trainable parameters
        self.name = name        # Layer identifier

    def forward(self, x):
        """
        Forward pass: compute layer output from input.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def backward(self, grad_output):
        """
        Backward pass: compute gradients.

        Args:
            grad_output: Gradient of loss w.r.t. layer output

        Returns:
            Gradient of loss w.r.t. layer input

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement backward()")


class Dense(Layer):
    """
    Fully connected (dense/linear) layer.

    Implements the transformation: output = input @ weights + bias

    Attributes:
        weights (Parameter): Weight matrix of shape (input_dim, output_dim)
        bias (Parameter): Bias vector of shape (1, output_dim)
    """

    def __init__(self, input_dimension, output_dimension, name=""):
        """
        Initialize dense layer with random weights and biases.

        Weights are initialized using scaled random normal distribution
        for better gradient flow during training.

        Args:
            input_dimension: Number of input features
            output_dimension: Number of output features
            name: Optional identifier for the layer
        """
        super().__init__(name)

        # Initialize bias and weights using random normal distribution
        # Scale by 1/sqrt(N) where N is the number of elements
        self.bias = Parameter(
            np.random.randn(1, output_dimension) / output_dimension**0.5
        )
        self.weights = Parameter(
            np.random.randn(input_dimension, output_dimension) /
            (output_dimension * input_dimension)**0.5
        )
        self.parameters = [self.bias, self.weights]

    def forward(self, x):
        """
        Forward pass: compute linear transformation.

        Args:
            x: Input array of shape (batch_size, input_dimension)

        Returns:
            Output array of shape (batch_size, output_dimension)
        """
        output = x @ self.weights + self.bias
        self.saved_arrays = [x]
        return output

    def backward(self, grad_output):
        """
        Backward pass: compute gradients for weights, bias, and input.

        Updates self.weights.gradient and self.bias.gradient.

        Args:
            grad_output: Gradient of loss w.r.t. output,
                        shape (batch_size, output_dimension)

        Returns:
            Gradient of loss w.r.t. input, shape (batch_size, input_dimension)
        """
        self.weights.zero_gradient()
        self.bias.zero_gradient()

        x = self.saved_arrays[0]

        # Compute gradients
        grad_input = grad_output @ self.weights.T
        self.weights.gradient = x.T @ grad_output
        self.bias.gradient = np.sum(grad_output, axis=0, keepdims=True)

        self.saved_arrays = []
        return grad_input


class Sigmoid(Layer):
    """
    Sigmoid activation layer.

    Applies the sigmoid function: σ(x) = 1 / (1 + exp(-x))
    Clips input to [-25, 25] to prevent numerical overflow.
    """

    def forward(self, x):
        """
        Forward pass: apply sigmoid activation.

        Args:
            x: Input array of shape (batch_size, features)

        Returns:
            Activated output of same shape as input
        """
        # Clip to avoid numerical overflow
        x_clipped = np.clip(x, -25.0, 25.0)
        sigmoid = 1.0 / (1.0 + np.exp(-x_clipped))
        self.saved_arrays = [sigmoid]
        return sigmoid

    def backward(self, grad_output):
        """
        Backward pass: compute gradient using sigmoid derivative.

        The derivative is: dσ/dx = σ(x) * (1 - σ(x))

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        sigmoid = self.saved_arrays[0]
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        self.saved_arrays = []
        return grad_input


class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) activation layer.

    Applies the ReLU function: ReLU(x) = max(0, x)
    """

    def forward(self, x):
        """
        Forward pass: apply ReLU activation.

        Args:
            x: Input array of shape (batch_size, features)

        Returns:
            Activated output of same shape as input
        """
        relu = np.maximum(0, x)
        self.saved_arrays = [x]
        return relu

    def backward(self, grad_output):
        """
        Backward pass: compute gradient using ReLU derivative.

        The derivative is: dReLU/dx = 1 if x > 0, else 0

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        x = self.saved_arrays[0]
        grad_input = grad_output * (x > 0)
        self.saved_arrays = []
        return grad_input
