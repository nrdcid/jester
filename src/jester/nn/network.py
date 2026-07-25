"""
Neural network container class.

This module provides the Network class that manages the stack of layers,
handles forward and backward passes, and coordinates training.
"""


class Network(object):
    """
    Neural network that chains multiple layers together.

    The Network class manages a sequential stack of layers, coordinates
    forward and backward passes through all layers, and integrates with
    optimizers and loss functions for training.

    Attributes:
        optimizer: Optimizer instance (e.g., SGD, Adam) for parameter updates
        loss: Loss function instance (e.g., MSE) for computing training loss
        layers (list): Sequential stack of Layer instances
    """

    def __init__(self, optimizer, loss):
        """
        Initialize a neural network.

        Args:
            optimizer: Optimizer instance for updating parameters
            loss: Loss function instance for computing loss and gradients
        """
        self.optimizer = optimizer
        self.loss = loss
        self.layers = []
        self.optimizer.set_layers(self.layers)

    def add_layer(self, layer):
        """
        Add a layer to the network.

        Layers are executed in the order they are added during forward pass,
        and in reverse order during backward pass.

        Args:
            layer: Layer instance to add to the network
        """
        self.layers.append(layer)

    def predict(self, x):
        """
        Make predictions on input data.

        Alias for forward pass, typically used for inference.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass through all layers.

        Sequentially applies each layer's forward method to transform
        the input into the final output.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size, output_dim)
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        """
        Backward pass through all layers.

        Performs backpropagation by:
        1. Getting initial gradient from loss function
        2. Propagating gradients backward through layers in reverse order
        3. Each layer computes parameter gradients and input gradients

        Note: Should only be called after a forward pass, as layers
        cache values needed for gradient computation.
        """
        # Get gradient from loss
        grad_output = self.loss.backward()

        # Propagate gradients backward through layers
        for layer in self.layers[::-1]:
            grad_output = layer.backward(grad_output)

    def fit(self, X, y, batch_size=32, epoch=1):
        """
        Train the network on a batch of data.

        Performs one training step:
        1. Forward pass to compute predictions
        2. Compute loss
        3. Backward pass to compute gradients
        4. Apply optimizer to update parameters

        Args:
            X: Input batch of shape (batch_size, input_dim)
            y: Target batch of shape (batch_size, output_dim)
            batch_size: Batch size (currently unused, for API compatibility)
            epoch: Number of epochs (currently unused, for API compatibility)

        Returns:
            Scalar loss value for this batch
        """
        # Forward pass
        output = self.forward(X)

        # Compute loss
        loss = self.loss.forward(output, y)

        # Backward pass
        self.backward()

        # Update parameters
        self.optimizer.apply_gradients()

        return loss
