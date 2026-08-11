import numpy as np
import pytest
from sklearn.datasets import make_classification

from jester.trees.tree import DecisionTree


@pytest.fixture
def data():
    return make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )


def test_default_criterion_is_gini(data):
    assert DecisionTree().criterion == "gini"


@pytest.mark.parametrize("criterion", ["gini", "entropy", "misclassification"])
def test_every_criterion_trains(data, criterion):
    X, y = data
    tree = DecisionTree(max_depth=4, criterion=criterion).fit(X, y)
    assert tree.score(X, y) > 0.6


def test_unknown_criterion_raises(data):
    X, y = data
    with pytest.raises(ValueError, match="unknown criterion"):
        DecisionTree(criterion="nonsense").fit(X, y)


def test_criterion_changes_fitted_tree(data):
    X, y = data
    gini = DecisionTree(max_depth=3, criterion="gini").fit(X, y).predict(X)
    misc = DecisionTree(max_depth=3, criterion="misclassification").fit(X, y).predict(X)
    assert not np.array_equal(gini, misc)
