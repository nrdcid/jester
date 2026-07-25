# Jester: ML Research Sandbox

Jester is an empirical research sandbox for defining, benchmarking, and statistically comparing machine learning algorithm variants.

Rather than solely implementing algorithms from scratch, Jester focuses on **generating empirical findings**: exposing algorithmic knobs (e.g., split criteria, distance metrics, weight update rules), executing multi-seed cross-validation sweeps across standard datasets, and comparing variants with statistical rigor.

---

## Core Architecture

The repository is structured into three distinct layers:

1. **Modular Algorithms (`src/jester/`)**
   Custom ML estimators refactored with explicit injection points for algorithmic knobs (split criteria, distance metrics, weak learner factories, shrinkage rates) while preserving backwards-compatible defaults.

2. **Research Harness (`src/jester/research/`)**
   - **Variant Registry (`registry.py`)**: Register named zero-arg estimator factories.
   - **Dataset Suite (`datasets.py`)**: Standardized benchmark datasets returning uniform `(X, y)` tuples.
   - **CV Harness (`harness.py`)**: Multi-seed stratified cross-validation capturing unaggregated per-fold metrics and runtime.
   - **Statistical Reporter (`report.py`)**: Results summarizer and paired significance tester using the Wilcoxon signed-rank test.

3. **Experiment Suite (`experiments/`)**
   Self-contained directories (`EXP-XXX-...`) tracking specific research questions:
   - `hypothesis.md`: Stated questions, hypotheses, and falsification criteria.
   - `run.py`: Reproducible execution script.
   - `results/raw.jsonl`: Committed raw per-fold results.
   - Appended empirical findings and statistical comparison tables.

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

---

## Tech Stack

- **Core**: Python, NumPy, pandas, SciPy, scikit-learn
- **Environment & Testing**: `uv`, `pytest`

