import numpy as np
import pytest

from jester.trees.metrics import CRITERIA, entropy, gini, impurity_reduction, misclassification


def test_gini_pure_node_is_zero():
    assert gini(np.array([1, 1, 1])) == pytest.approx(0.0)


def test_gini_balanced_binary_is_half():
    assert gini(np.array([0, 0, 1, 1])) == pytest.approx(0.5)


def test_gini_handles_negative_and_string_labels():
    assert gini(np.array([-1, -1, 1, 1])) == pytest.approx(0.5)
    assert gini(np.array(["cat", "cat", "dog", "dog"])) == pytest.approx(0.5)


def test_other_criteria():
    y = np.array([0, 0, 1, 1])
    assert entropy(y) == pytest.approx(1.0)
    assert misclassification(y) == pytest.approx(0.5)


def test_empty_node_is_zero_for_all_criteria():
    for criterion in CRITERIA.values():
        assert criterion(np.array([], dtype=int)) == pytest.approx(0.0)


def test_criterion_is_injectable():
    y = np.array([0, 0, 1, 1])
    left, right = np.array([0, 1]), np.array([2, 3])
    assert impurity_reduction(y, left, right) == pytest.approx(0.5)
    assert impurity_reduction(y, left, right, criterion=entropy) == pytest.approx(1.0)
