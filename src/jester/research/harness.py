"""Cross-validated measurement loop retaining one row per fold."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from jester.metrics import accuracy
from .datasets import load


@dataclass(frozen=True)
class Result:
    variant: str
    dataset: str
    seed: int
    fold: int
    accuracy: float
    fit_seconds: float
    predict_seconds: float
    n_train: int
    n_test: int
    error: str = ""


def run_experiment(variants, datasets, seeds=(0, 1, 2, 3, 4), n_folds=5,
                   verbose=False):
    """Evaluate every factory across every dataset, seed, and fold."""
    results = []
    for dataset_name in datasets:
        X, y = load(dataset_name)
        for seed in seeds:
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
            for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                for name, factory in variants.items():
                    result = _run_one_fold(
                        name, factory, dataset_name, seed, fold,
                        X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                    )
                    results.append(result)
                    if verbose:
                        print(f"  {dataset_name} seed={seed} fold={fold} {name}")
    return results


def _run_one_fold(name, factory, dataset_name, seed, fold,
                  X_train, y_train, X_test, y_test):
    common = {
        "variant": name, "dataset": dataset_name, "seed": seed, "fold": fold,
        "n_train": len(y_train), "n_test": len(y_test),
    }
    try:
        model = factory()
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - started
        return Result(
            accuracy=accuracy(y_test, y_pred), fit_seconds=fit_seconds,
            predict_seconds=predict_seconds, **common,
        )
    except Exception as exc:  # experiment failures are data, not aborts
        return Result(
            accuracy=float("nan"), fit_seconds=float("nan"),
            predict_seconds=float("nan"), error=f"{type(exc).__name__}: {exc}",
            **common,
        )


def save_results(results, path):
    """Write results as newline-delimited JSON."""
    with open(path, "w") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")


def load_results(path):
    """Read results written by :func:`save_results`."""
    with open(path) as handle:
        return [Result(**json.loads(line)) for line in handle if line.strip()]
