import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from jester.ensemble.bagging import BaggingEnsemble


@pytest.fixture
def data():
    return make_classification(
        n_samples=150, n_features=5, n_informative=3, n_redundant=0,
        n_classes=2, random_state=0,
    )


def test_fit_does_not_touch_global_rng(data):
    X, y = data
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    BaggingEnsemble(n_estimators=3, random_state=7).fit(X, y)
    after = np.random.random()
    assert before == pytest.approx(after)


def test_same_seed_is_reproducible(data):
    X, y = data
    a = BaggingEnsemble(n_estimators=5, random_state=0).fit(X, y).predict(X)
    b = BaggingEnsemble(n_estimators=5, random_state=0).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_base_learner_is_injectable(data):
    X, y = data
    calls = []

    def factory():
        calls.append(1)
        return DecisionTreeClassifier(max_depth=1)

    BaggingEnsemble(
        n_estimators=4, base_learner=factory, random_state=0
    ).fit(X, y)
    assert len(calls) == 4
