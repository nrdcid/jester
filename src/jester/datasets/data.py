"""
Dataset utilities for loading and managing training data.

This module provides dataset classes for the circles classification task
and IMDB sentiment analysis.
"""

import os
import json
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


class Circles(object):
    """
    Concentric circles dataset for binary classification.

    Generates a toy dataset with two classes arranged in concentric circles,
    useful for testing non-linear classifiers.

    Attributes:
        X (np.ndarray): Feature matrix of shape (n_samples, 2)
        labels (np.ndarray): Binary labels (0/1 or -1/1)
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Testing labels

    Example:
        >>> circles = Circles()
        >>> print(circles.X_train.shape, circles.y_train.shape)
        (280, 2) (280,)
    """

    def __init__(self, mode="0/1"):
        """
        Initialize circles dataset.

        Args:
            mode: Label encoding scheme
                  "0/1" for binary labels 0 and 1 (default)
                  "-1/1" for binary labels -1 and 1
        """
        self.X, self.labels = make_circles(
            n_samples=400,
            noise=0.11,
            random_state=2025,
            factor=0.78
        )

        if mode == "-1/1":
            self.labels = self.labels * 2 - 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.labels,
            test_size=0.3,
            random_state=46222025
        )


class IMDB:
    """
    IMDB movie review sentiment dataset.

    Loads movie reviews with sentiment labels (positive/negative) from
    a JSON file for binary sentiment classification.

    Attributes:
        data (dict): Raw dataset loaded from JSON
        X_train (list): Training text samples
        X_test (list): Testing text samples
        y_train (list): Training labels (0=negative, 1=positive)
        y_test (list): Testing labels

    Example:
        >>> imdb = IMDB()
        >>> print(len(imdb.X_train), len(imdb.y_train))
        1400 1400
    """

    def __init__(self, data_path=None):
        """
        Initialize IMDB dataset.

        Args:
            data_path: Optional path to movie_review_data.json.
                      If None, searches in standard locations.
        """
        if data_path is None:
            # Try to find the data file in common locations
            current_folder = os.path.dirname(os.path.abspath(__file__))
            parent_folder = os.path.dirname(os.path.dirname(current_folder))

            possible_paths = [
                os.path.join(parent_folder, "data", "movie_review_data.json"),
                os.path.join(current_folder, "movie_review_data.json"),
                "movie_review_data.json"
            ]

            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break

            if data_path is None:
                raise FileNotFoundError(
                    "Could not find movie_review_data.json. "
                    "Please provide the path explicitly."
                )

        with open(data_path) as f:
            self.data = json.load(f)

        X = [d['text'] for d in self.data['data']]
        y = [d['label'] for d in self.data['data']]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.3,
            shuffle=True,
            random_state=46222025
        )
