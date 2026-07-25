"""
Evaluation utilities for KNN models.

This module contains functions for evaluating and preparing data
for KNN classifiers and regressors.
"""

import numpy as np
import matplotlib.pyplot as plt
from jester.viz import display_confusion
from .classifiers import KNNClassifier


def prepare_data(dataset):
    """
    Prepare dataset for KNN by reshaping features and reporting statistics.

    For image datasets (like MNIST), this function reshapes the 3D arrays
    (n_samples, height, width) into 2D arrays (n_samples, height*width)
    suitable for KNN algorithms.

    Parameters
    ----------
    dataset :
        Dataset object with X_train, X_valid, X_test arrays

    Returns
    -------
    dataset :
        Modified dataset with reshaped arrays

    Side Effects
    ------------
    Prints dataset statistics including:
    - Original and reshaped array shapes
    - Number of pixels per datapoint
    - Size of training, validation, and test sets
    """
    print(f"Original Training Shape: {dataset.X_train.shape}")
    print(f"Original Validation Shape: {dataset.X_valid.shape}")
    print(f"Original Test Shape: {dataset.X_test.shape}")

    n_train, height, width = dataset.X_train.shape
    n_valid = dataset.X_valid.shape[0]
    n_test = dataset.X_test.shape[0]

    dataset.X_train = dataset.X_train.reshape(n_train, height * width)
    dataset.X_valid = dataset.X_valid.reshape(n_valid, height * width)
    dataset.X_test = dataset.X_test.reshape(n_test, height * width)

    print(f"Datapoint Size (pixels): {height * width}")
    print(f"Training Size: {len(dataset.X_train)}")
    print(f"Validation Size: {len(dataset.X_valid)}")
    print(f"Test Size: {len(dataset.X_test)}")
    print(f"Total: {len(dataset.X_train) + len(dataset.X_valid) + len(dataset.X_test)}")

    print(f"Reshaped Training Shape: {dataset.X_train.shape}")
    print(f"Reshaped Validation Shape: {dataset.X_valid.shape}")
    print(f"Reshaped Test Shape: {dataset.X_test.shape}")

    return dataset


def evaluate(ks_range, dataset, KNNClass=KNNClassifier):
    """
    Evaluate KNN model across different values of k.

    This function performs model selection by:
    1. Training models with different k values on the training set
    2. Evaluating each on the validation set
    3. Selecting the best k based on validation accuracy
    4. Reporting final test set performance

    Parameters
    ----------
    ks_range : iterable
        Range of k values to evaluate
    dataset :
        Dataset with train/valid/test splits
    KNNClass : class, default=KNNClassifier
        The KNN class to use (KNNClassifier or WeightedKNNClassifier)

    Side Effects
    ------------
    Prints:
    - Best k value on validation set
    - Accuracy on test set
    - Confusion matrix visualization on test set
    """
    accuracy = 0
    ks = ks_range
    accuracies_valid = []

    for k in ks:
        print(k, end="\r")
        knn = KNNClass(k).fit(dataset.X_train, dataset.y_train)
        acc_valid = knn.accuracy(dataset.X_valid, dataset.y_valid)
        accuracies_valid.append(acc_valid)

        if acc_valid > accuracy:
            accuracy = acc_valid
            best_valid_k = k

    print("Best k Validation:", best_valid_k)

    knn = KNNClass(best_valid_k).fit(dataset.X_train, dataset.y_train)
    acc_test = knn.accuracy(dataset.X_test, dataset.y_test)

    print("Accuracy Test Set:", acc_test)
    display_confusion(knn.confusion_matrix(dataset.X_test, dataset.y_test))


def plot_k_vs_metric(ks, train_metrics, valid_metrics, metric_name="Accuracy",
                     ylabel="Accuracy", title=None):
    """
    Plot training and validation metrics versus k values.

    Parameters
    ----------
    ks : list
        List of k values
    train_metrics : list
        Metrics computed on training set
    valid_metrics : list
        Metrics computed on validation set
    metric_name : str, default="Accuracy"
        Name of the metric for legend
    ylabel : str, default="Accuracy"
        Label for y-axis
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, valid_metrics, label=f"Validation {metric_name}", marker='o')
    ax.plot(ks, train_metrics, label=f"Training {metric_name}", marker='s')
    ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{metric_name} vs k", fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
