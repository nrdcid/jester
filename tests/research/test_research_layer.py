import numpy as np
import pytest

from jester.research.datasets import DATASETS, load
from jester.research.harness import Result, run_experiment
from jester.research.registry import _REGISTRY, get_variant, list_variants, variant


def test_dataset_contract():
    assert len(DATASETS) >= 4
    for name in DATASETS:
        X, y = load(name)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        assert np.isfinite(X).all()


def test_dataset_is_deterministic():
    first = load("circles")
    second = load("circles")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_registry_lifecycle():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        @variant("b")
        def b():
            return 2

        variant("a")(lambda: 1)
        assert get_variant("b")() == 2
        assert list_variants() == ["a", "b"]
        with pytest.raises(ValueError, match="already registered"):
            variant("a")(lambda: 3)
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


class Constant:
    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.label = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.label)


def test_harness_preserves_provenance_and_records_failures():
    results = run_experiment(
        {"constant": Constant}, ["circles"], seeds=[0], n_folds=2
    )
    assert len(results) == 2
    assert isinstance(results[0], Result)
    assert results[0].variant == "constant"

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("boom")

    failed = run_experiment({"broken": Broken}, ["circles"], seeds=[0], n_folds=2)
    assert all(np.isnan(result.accuracy) for result in failed)
    assert all("boom" in result.error for result in failed)
