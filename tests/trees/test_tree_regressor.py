import numpy as np
import pytest

from jester.trees.tree import DecisionTreeRegressor


def test_regressor_fits_piecewise_constant_function():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    tree = DecisionTreeRegressor(max_depth=1).fit(X, y)

    assert tree.predict(X) == pytest.approx(y)
    assert tree.score(X, y) == pytest.approx(1.0)


def test_regressor_predicts_mean_at_max_depth_zero():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 8.0])

    tree = DecisionTreeRegressor(max_depth=0).fit(X, y)

    assert tree.predict(np.array([[-1.0], [4.0]])) == pytest.approx([4.0, 4.0])


def test_regressor_feature_importance_uses_variance_reduction():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    tree = DecisionTreeRegressor(max_depth=1).fit(X, y)

    assert tree.feature_importance(X, y) == {0: pytest.approx(1.0)}
