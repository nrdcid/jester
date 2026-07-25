"""
KNN Regressor implementation for regression tasks.

This module contains the K-Nearest Neighbors regressor implementation
that predicts continuous values by averaging the target values of
the k nearest neighbors.
"""

import numpy as np
from .classifiers import KNNClassifier


class KNNRegressor(KNNClassifier):
    """
    K-Nearest Neighbors Regressor.

    Predicts continuous target values by averaging the target values
    of the k nearest neighbors from the training data.

    Inherits from KNNClassifier and overrides the voting mechanism
    to perform averaging instead of majority voting.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use for regression.

    Attributes
    ----------
    _k : int
        Number of neighbors
    _ball_tree : sklearn.neighbors.BallTree
        Tree structure for efficient nearest neighbor queries
    _y : np.ndarray
        Training target values
    """

    def average_vote(self, indices_nearest_k, distances_nearest_k=None):
        """
        Calculate average target value from nearest neighbors.

        Parameters
        ----------
        indices_nearest_k : np.ndarray of shape (n_queries, k)
            Indices of k nearest neighbors for each query point
        distances_nearest_k : np.ndarray of shape (n_queries, k), optional
            Distances to k nearest neighbors (not used in basic averaging)

        Returns
        -------
        voted_labels : np.ndarray of shape (n_queries,)
            Predicted target values (average of k nearest neighbors)
        """
        voted_labels = np.empty(indices_nearest_k.shape[0])

        for i in range(indices_nearest_k.shape[0]):
            neighbor_indices = indices_nearest_k[i]
            neighbor_values = self._y[neighbor_indices]
            voted_labels[i] = np.mean(neighbor_values)

        return voted_labels

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query samples

        Returns
        -------
        y_pred : np.ndarray of shape (n_queries,)
            Predicted target values
        """
        distances_nearest_k, indices_nearest_k = self._ball_tree.query(X, k=self._k)
        return self.average_vote(indices_nearest_k, distances_nearest_k)

    def mse(self, X, y):
        """
        Calculate Mean Squared Error on test data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True target values

        Returns
        -------
        score : float
            Mean squared error between predictions and true values
        """
        y_pred = self.predict(X)
        squared_errors = (y - y_pred) ** 2
        score = np.mean(squared_errors)
        return score
