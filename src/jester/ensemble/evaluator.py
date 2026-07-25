"""
Model evaluation utilities for comparing ensemble methods.
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


class Evaluator:
    """
    Test multiple model performance using cross-validation.

    Evaluates models using stratified k-fold cross-validation and tracks
    both performance metrics and execution time.
    """

    def __init__(self, dataset, n_folds=3):
        """
        Initialize the evaluator.

        Args:
            dataset: Dataset object with X and y attributes
            n_folds: Number of folds for cross-validation
        """
        self.dataset = dataset
        self.execution_time = {}  # model name -> execution time
        self.scores = {}  # model name -> average accuracy
        self.score_name = 'WAP'  # Weighted Average Precision (accuracy)
        self.k_fold = StratifiedKFold(n_folds, shuffle=True, random_state=42)

    def evaluate_model(self, model, name):
        """
        Fit the model and evaluate using k-fold cross-validation.

        Args:
            model: Classifier with fit and predict methods
            name: Name identifier for this model
        """
        start = time()
        self.scores[name] = 0

        # Perform k-fold cross-validation
        for train_index, test_index in self.k_fold.split(self.dataset.X, self.dataset.y):
            X_train, X_test = self.dataset.X[train_index], self.dataset.X[test_index]
            Y_train, Y_test = self.dataset.y[train_index], self.dataset.y[test_index]

            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            self.scores[name] += np.mean(Y_test == Y_pred)

        # Calculate average score across folds
        self.scores[name] /= self.k_fold.n_splits

        self.execution_time[name] = time() - start

    def print_results(self):
        """
        Print results for all evaluated models in a formatted table.
        """
        models_cross = pd.DataFrame({
            'Model': list(self.scores.keys()),
            self.score_name: list(self.scores.values()),
            'Execution time': list(self.execution_time.values())
        })
        print(models_cross.sort_values(by=self.score_name, ascending=False))

    def plot_metrics(self):
        """
        Plot bar charts comparing model performance and execution time.
        """
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(6)
        fig.set_figwidth(18)

        # Plot scores
        left = [i for i in range(len(self.scores))]
        height = list(self.scores.values())
        tick_label = list(self.scores.keys())
        axs[0].set_title(self.score_name)
        axs[0].bar(left, height, tick_label=tick_label, width=0.5)

        # Plot execution times
        height = list(self.execution_time.values())
        tick_label = list(self.execution_time.keys())
        axs[1].set_title("Elapsed time")
        axs[1].bar(left, height, tick_label=tick_label, width=0.5)

        plt.show()
