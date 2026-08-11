# Jester: ML Research Sandbox

Jester is a research sandbox for turning handwritten machine-learning
implementations into empirical findings. Algorithmic choices such as tree
criteria, KNN distance metrics, and ensemble learning rates are explicit
variant knobs. The research layer evaluates those variants on fixed datasets
with matched cross-validation folds and paired statistical comparisons.

---

## Core Architecture

The repository is structured into three distinct layers:

1. **Modular Algorithms (`src/jester/`)**
   Custom ML estimators refactored with explicit injection points for algorithmic knobs (split criteria, distance metrics, weak learner factories, shrinkage rates) while preserving backwards-compatible defaults.

2. **Research Harness (`src/jester/research/`)**
   - **Variant Registry**: Register named zero-argument estimator factories.
   - **Dataset Suite**: Standardized benchmark datasets returning `(X, y)`.
   - **CV Harness**: Multi-seed stratified cross-validation with per-fold
     accuracy and timing.
   - **Statistical Reporter**: Summary tables and paired Wilcoxon tests.

3. **Experiment Suite (`experiments/`)**
   Self-contained directories (`EXP-XXX-...`) tracking specific research questions:
   - `hypothesis.md`: Stated questions, hypotheses, and falsification criteria.
   - `run.py`: Reproducible execution script.
   - `results/raw.jsonl`: Committed raw per-fold results.
   - Appended empirical findings and statistical comparison tables.

The first experiment is [`EXP-001`](experiments/EXP-001-split-criteria/), which
compares decision-tree split criteria.

---

## Quickstart

### Environment Setup

Install dependencies using `uv`:

```bash
uv sync --extra dev
```

### Running Tests

```bash
uv run pytest
```

### Running an Experiment

Execute an experiment sweep:

```bash
uv run python experiments/EXP-001-split-criteria/run.py
```

### Implemented algorithms

Decision trees, KNN classifiers/regressors, logistic and ridge regression,
bagging, random forests, AdaBoost/SAMME, a NumPy MLP with handwritten
backpropagation, a vanilla RNN, perceptron, SVM kernels, and an optional
PyTorch autoencoder are available under `src/jester/`.

---

## Tech Stack

- **Core**: Python, NumPy, pandas, SciPy, scikit-learn
- **Environment & Testing**: `uv`, `pytest`

