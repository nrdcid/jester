"""
Optimization algorithms for training neural networks.

This module provides optimizer implementations including SGD (Stochastic Gradient Descent)
and Adam (Adaptive Moment Estimation).
"""

import numpy as np


class SGD(object):
    """
    Stochastic Gradient Descent optimizer.

    Performs parameter updates using the rule:
    θ = θ - learning_rate * gradient

    Attributes:
        learning_rate (float): Step size for parameter updates
        layers (list): Reference to network layers containing parameters
    """

    def __init__(self, learning_rate):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Step size for gradient descent updates
        """
        self.learning_rate = learning_rate
        self.layers = None

    def set_layers(self, layers):
        """
        Store reference to network layers.

        Args:
            layers: List of Layer instances from the network
        """
        self.layers = layers

    def apply_gradients(self):
        """
        Apply gradients to update all parameters.

        Loops through all layers and their parameters, applying:
        parameter = parameter - learning_rate * gradient
        """
        for layer in self.layers:
            for p in layer.parameters:
                p.apply_gradient(-self.learning_rate * p.gradient)


class Adam(SGD):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum with adaptive learning rates for each parameter.
    Uses moving averages of gradients (first moment) and squared gradients
    (second moment) with bias correction.

    Algorithm:
        t = t + 1
        m = β₁ * m + (1 - β₁) * g
        v = β₂ * v + (1 - β₂) * g²
        m_hat = m / (1 - β₁ᵗ)
        v_hat = v / (1 - β₂ᵗ)
        θ = θ - α * m_hat / (√v_hat + ε)

    Reference: https://arxiv.org/pdf/1412.6980.pdf

    Attributes:
        learning_rate (float): Step size (α)
        beta_1 (float): Exponential decay rate for first moment (default: 0.9)
        beta_2 (float): Exponential decay rate for second moment (default: 0.999)
        first_moment (dict): First moment estimates for each parameter
        second_moment (dict): Second moment estimates for each parameter
        time_step (int): Current iteration count
    """

    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Step size for updates (α)
            beta_1: Decay rate for first moment estimates
            beta_2: Decay rate for second moment estimates
        """
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_moment = None
        self.second_moment = None
        self.time_step = 0

    def apply_gradients(self):
        """
        Apply gradients using Adam update rule.

        Updates parameters using adaptive learning rates based on
        first and second moment estimates with bias correction.
        """
        self.time_step += 1
        epsilon = 1e-8

        # Initialize moment dictionaries on first call
        if self.first_moment is None:
            self.first_moment = {}
            self.second_moment = {}
            for layer in self.layers:
                for p in layer.parameters:
                    self.first_moment[id(p)] = np.zeros_like(p)
                    self.second_moment[id(p)] = np.zeros_like(p)

        # Update each parameter
        for layer in self.layers:
            for p in layer.parameters:
                g = p.gradient
                m = self.first_moment[id(p)]
                v = self.second_moment[id(p)]

                # Update biased first and second moments
                m = self.beta_1 * m + (1 - self.beta_1) * g
                v = self.beta_2 * v + (1 - self.beta_2) * (g ** 2)

                # Store updated moments
                self.first_moment[id(p)] = m
                self.second_moment[id(p)] = v

                # Bias correction
                m_hat = m / (1 - self.beta_1 ** self.time_step)
                v_hat = v / (1 - self.beta_2 ** self.time_step)

                # Apply update
                p.apply_gradient(
                    -self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                )
