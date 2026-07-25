"""
KNN - K-Nearest Neighbors Implementation

A production-ready implementation of K-Nearest Neighbors algorithms
for both classification and regression tasks.

Main Classes
------------
KNNClassifier : Standard KNN classifier using majority voting
WeightedKNNClassifier : Distance-weighted KNN classifier
KNNRegressor : KNN regressor using average of neighbors

Utilities
---------
evaluate : Model selection and evaluation across different k values
prepare_data : Data preparation for image datasets
plot_k_vs_metric : Visualization of performance vs k

Examples
--------
>>> from jester.knn import KNNClassifier
>>>
>>> clf = KNNClassifier(k=3)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
>>> accuracy = clf.accuracy(X_test, y_test)
"""

from .classifiers import KNNClassifier, WeightedKNNClassifier
from .regressors import KNNRegressor
from .evaluation import evaluate, prepare_data, plot_k_vs_metric

__all__ = [
    'KNNClassifier',
    'WeightedKNNClassifier',
    'KNNRegressor',
    'evaluate',
    'prepare_data',
    'plot_k_vs_metric',
]

__version__ = '1.0.0'
