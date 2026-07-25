"""
MLP from Scratch - A minimal neural network framework.

This package provides a simple implementation of multi-layer perceptrons (MLPs)
built from scratch using only NumPy. It includes:

- Layers: Dense, Sigmoid, ReLU
- Loss functions: MSE
- Optimizers: SGD, Adam
- Network: Container for building and training models
- Parameter: Trainable parameters with gradient tracking

Example:
    >>> from jester.nn import Network, Dense, ReLU, Sigmoid, MSE, SGD
    >>>
    >>> # Build a simple network
    >>> network = Network(optimizer=SGD(learning_rate=0.1), loss=MSE())
    >>> network.add_layer(Dense(2, 16))
    >>> network.add_layer(ReLU())
    >>> network.add_layer(Dense(16, 1))
    >>> network.add_layer(Sigmoid())
    >>>
    >>> # Train the network
    >>> loss = network.fit(X_train, y_train)
    >>> predictions = network.predict(X_test)
"""

from .parameter import Parameter
from .layers import Layer, Dense, Sigmoid, ReLU
from .losses import MSE
from .optimizers import SGD, Adam
from .network import Network

__all__ = [
    'Parameter',
    'Layer',
    'Dense',
    'Sigmoid',
    'ReLU',
    'MSE',
    'SGD',
    'Adam',
    'Network',
]

__version__ = '0.1.0'
