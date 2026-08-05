"""Small deterministic benchmark datasets with a uniform ``(X, y)`` API."""

from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

_SEED = 2025


def _circles():
    return skd.make_circles(
        n_samples=400, noise=0.11, factor=0.78, random_state=_SEED
    )


def _moons():
    return skd.make_moons(n_samples=400, noise=0.25, random_state=_SEED)


def _blobs_overlapping():
    return skd.make_blobs(
        n_samples=400, centers=3, cluster_std=2.6, n_features=4,
        random_state=_SEED,
    )


def _classification_noisy():
    return skd.make_classification(
        n_samples=400, n_features=10, n_informative=5, n_redundant=0,
        n_repeated=0, n_classes=3, flip_y=0.05, random_state=_SEED,
    )


def _wine():
    data = skd.load_wine()
    return data.data, data.target


def _breast_cancer():
    data = skd.load_breast_cancer()
    return data.data, data.target


DATASETS = {
    "circles": _circles,
    "moons": _moons,
    "blobs_overlapping": _blobs_overlapping,
    "classification_noisy": _classification_noisy,
    "wine": _wine,
    "breast_cancer": _breast_cancer,
}


def load(name):
    """Load ``name`` and return finite float features plus 1-D labels."""
    try:
        loader = DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}"
        ) from None
    X, y = loader()
    return np.asarray(X, dtype=float), np.asarray(y).reshape(-1)
