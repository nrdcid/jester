"""
KNN Classifier implementations for classification tasks.

This module contains implementations of K-Nearest Neighbors classifiers:
- KNNClassifier: Standard KNN classifier using majority voting
- WeightedKNNClassifier: Distance-weighted KNN classifier
"""

import numpy as np
import sklearn.neighbors


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier using majority voting.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use for classification.

    Attributes
    ----------
    _k : int
        Number of neighbors
    _ball_tree : sklearn.neighbors.BallTree
        Tree structure for efficient nearest neighbor queries
    _y : np.ndarray
        Training labels
    label_to_index : dict
        Mapping from class labels to indices
    index_to_label : dict
        Mapping from indices to class labels
    training_most_common : int or float
        Most common label in training set (used for tie-breaking)
    """

    def __init__(self, k=5):
        self._k = k
        self._ball_tree = None
        self._y = None
        self.label_to_index = None
        self.index_to_label = None
        self.training_most_common = None

    def fit(self, X, y):
        """
        Fit the KNN classifier using training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data features
        y : np.ndarray of shape (n_samples,)
            Training data labels

        Returns
        -------
        self : KNNClassifier
            The fitted classifier
        """
        self._ball_tree = sklearn.neighbors.BallTree(X)
        self._y = np.asarray(y)
        classes = np.unique(self._y)
        self.label_to_index = dict(zip(classes, range(classes.shape[0])))
        self.index_to_label = dict(zip(range(classes.shape[0]), classes))
        label_values, label_counts = np.unique(self._y, return_counts=True)
        self.training_most_common = label_values[np.argmax(label_counts)]
        return self

    def sample_label(self, index):
        """
        Get the label of a training sample by index.

        Parameters
        ----------
        index : int
            Index of the training sample

        Returns
        -------
        label : int or float
            The label of the training sample
        """
        assert index < self._y.shape[0]
        return self._y[index]

    def majority_vote(self, indices_nearest_k, distances_nearest_k=None):
        """
        Determine majority class from nearest neighbors.

        Uses the most common class among k nearest neighbors.
        Ties are broken using the most common class in the training set.

        Parameters
        ----------
        indices_nearest_k : np.ndarray of shape (n_queries, k)
            Indices of k nearest neighbors for each query point
        distances_nearest_k : np.ndarray of shape (n_queries, k), optional
            Distances to k nearest neighbors (not used in basic majority vote)

        Returns
        -------
        voted_labels : np.ndarray of shape (n_queries,)
            Predicted labels for each query point
        """
        M = indices_nearest_k.shape[0]
        voted_labels = np.empty(M, dtype=self._y.dtype)
        C = len(self.label_to_index)
        tie_break_idx = self.label_to_index[self.training_most_common]

        for i in range(M):
            neighbor_labels = self._y[indices_nearest_k[i]]
            neighbor_idx = np.array([self.label_to_index[l] for l in neighbor_labels], dtype=int)
            counts = np.bincount(neighbor_idx, minlength=C)
            max_count = counts.max()
            tied = np.flatnonzero(counts == max_count)

            if tied.size == 1:
                winner_idx = tied[0]
            else:
                winner_idx = tie_break_idx

            voted_labels[i] = self.index_to_label[winner_idx]

        return voted_labels

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query samples

        Returns
        -------
        y_pred : np.ndarray of shape (n_queries,)
            Predicted class labels
        """
        distances_nearest_k, indices_nearest_k = self._ball_tree.query(X, k=self._k)
        return self.majority_vote(indices_nearest_k, distances_nearest_k)

    def confusion_matrix(self, X, y):
        """
        Compute confusion matrix for predictions on X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        c_matrix : np.ndarray of shape (n_classes, n_classes)
            Confusion matrix where c_matrix[i, j] is the count of samples
            with true label i that were predicted as label j
        """
        C = len(self.label_to_index)
        c_matrix = np.zeros((C, C), dtype=int)
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        y_true_idx = np.array([self.label_to_index[lab] for lab in y_true], dtype=int)
        y_pred_idx = np.array([self.label_to_index[lab] for lab in y_pred], dtype=int)

        for t, p in zip(y_true_idx, y_pred_idx):
            c_matrix[t, p] += 1

        return c_matrix

    def accuracy(self, X, y):
        """
        Calculate classification accuracy on test data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples
        y : np.ndarray of shape (n_samples,)
            True labels

        Returns
        -------
        accuracy : float
            Fraction of correctly classified samples
        """
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        score = float(np.mean(y_pred == y_true))
        return score


class WeightedKNNClassifier(KNNClassifier):
    """
    Weighted K-Nearest Neighbors Classifier.

    Similar to KNNClassifier but uses distance-weighted voting where
    closer neighbors have more influence on the prediction.

    The weight for each neighbor is proportional to the inverse of its
    distance from the query point.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to use for classification.
    """

    def weighted_vote(self, indices_nearest_k, distances_nearest_k):
        """
        Determine class using distance-weighted voting.

        Each neighbor votes for its class with a weight inversely proportional
        to its distance. If a query point has distance 0 to any training point,
        that training point's label is returned directly.

        Parameters
        ----------
        indices_nearest_k : np.ndarray of shape (n_queries, k)
            Indices of k nearest neighbors for each query point
        distances_nearest_k : np.ndarray of shape (n_queries, k)
            Distances to k nearest neighbors

        Returns
        -------
        labels : np.ndarray of shape (n_queries,)
            Predicted labels for each query point
        """
        labels = []

        for i in range(indices_nearest_k.shape[0]):
            neighbor_indices = indices_nearest_k[i]
            neighbor_distances = distances_nearest_k[i]
            neighbor_labels = self._y[neighbor_indices]

            # Handle exact matches (distance = 0)
            zero_distance_mask = neighbor_distances == 0
            if np.any(zero_distance_mask):
                exact_match_labels = neighbor_labels[zero_distance_mask]
                labels.append(exact_match_labels[0])
                continue

            # Compute distance-based weights
            weights = 1.0 / neighbor_distances

            # Calculate weighted votes for each class
            unique_classes = np.unique(neighbor_labels)
            weighted_votes = {}

            for class_label in unique_classes:
                class_mask = neighbor_labels == class_label
                weighted_votes[class_label] = np.sum(weights[class_mask])

            # Select class with highest weighted vote
            best_class = max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
            labels.append(best_class)

        return np.array(labels)

    def predict(self, X):
        """
        Predict class labels for samples in X using weighted voting.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query samples

        Returns
        -------
        labels : np.ndarray of shape (n_queries,)
            Predicted class labels
        """
        distances_nearest_k, indices_nearest_k = self._ball_tree.query(X, k=self._k)
        labels = self.weighted_vote(indices_nearest_k, distances_nearest_k)
        return labels
