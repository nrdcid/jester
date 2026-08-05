"""Infrastructure for measuring machine-learning algorithm variants."""

from .datasets import DATASETS, load
from .harness import Result, load_results, run_experiment, save_results
from .registry import get_variant, list_variants, variant
from .report import compare_to_baseline, summarize, to_markdown

__all__ = [
    "DATASETS", "load", "Result", "run_experiment", "save_results",
    "load_results", "variant", "get_variant", "list_variants", "summarize",
    "compare_to_baseline", "to_markdown",
]
