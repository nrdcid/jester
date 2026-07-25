"""Visualization utilities for classifiers and data inspection."""

import numpy as np
import matplotlib.pyplot as plt


def show_decision_surface(model, X, y, ax=None):
    """
    Visualize the decision surface of a trained classification model.

    Args:
        model: Trained model with a predict() method
        X: Feature matrix of shape (n_samples, 2)
        y: True labels of shape (n_samples,)
        ax: Optional matplotlib axis. If None, creates and shows a figure.

    Example:
        >>> from jester.nn import Network, Dense, ReLU, Sigmoid, MSE, SGD
        >>> from jester.datasets import Circles
        >>> from jester.viz import show_decision_surface
        >>>
        >>> circles = Circles()
        >>> network = Network(optimizer=SGD(0.1), loss=MSE())
        >>> # ... train network ...
        >>> show_decision_surface(network, circles.X, circles.labels)
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_grid, y_grid = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_grid, y_grid)

    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))

    y_hat = model.predict(grid).reshape(-1,)
    zz = y_hat.reshape(xx.shape)

    if ax is None:
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, cmap='PiYG', alpha=0.8)
        plt.colorbar(label='Predicted probability')
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='PiYG')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Surface')
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='PiYG')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Surface')


def display_confusion(c_matrix):
    """
    Display a confusion matrix.

    Args:
        c_matrix: square confusion matrix, shape (num_classes, num_classes)
    """
    _, ax = plt.subplots()
    ax.matshow(c_matrix, cmap=plt.cm.Blues)
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[0]):
            ax.text(i, j, str(c_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel("predicted label", fontsize=16)
    ax.set_ylabel("true label", fontsize=16)
    plt.show()


def plot_data(X, y):
    """
    Scatter-plot 2D binary data with labels expected as -1 or 1.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    colors = ["steelblue" if yi == -1 else "#a76c6e" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], color=colors, s=75)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.show()
