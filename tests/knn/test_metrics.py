import numpy as np
import pytest
from sklearn.datasets import make_blobs

from jester.knn import KNNClassifier


@pytest.fixture
def data():
    return make_blobs(n_samples=90, centers=3, n_features=4, random_state=0)


def test_default_metric_is_euclidean(data):
    assert KNNClassifier().metric == "euclidean"


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_builtin_metrics_fit_and_predict(data, metric):
    X, y = data
    assert KNNClassifier(k=3, metric=metric).fit(X, y).accuracy(X, y) > 0.8


def test_mahalanobis_metric_with_params(data):
    X, y = data
    vi = np.linalg.pinv(np.cov(X, rowvar=False))
    clf = KNNClassifier(k=3, metric="mahalanobis", metric_params={"VI": vi})
    assert clf.fit(X, y).accuracy(X, y) > 0.8


def test_metric_reaches_ball_tree(data):
    X, y = data
    clf = KNNClassifier(k=3, metric="manhattan").fit(X, y)
    distances, _ = clf._ball_tree.query(X[:1], k=1)
    assert distances[0][0] == pytest.approx(0.0)
