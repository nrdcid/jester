"""Aggregate experiment results and perform paired comparisons."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats


def to_frame(results):
    return pd.DataFrame([asdict(result) for result in results])


def summarize(results, metric="accuracy"):
    """Return one summary row per variant and dataset."""
    frame = to_frame(results)
    if frame.empty:
        return frame
    frame["_failed"] = frame[metric].isna()
    summary = frame.groupby(["variant", "dataset"], as_index=False).agg(
        **{
            f"{metric}_mean": (metric, "mean"),
            f"{metric}_std": (metric, "std"),
            f"{metric}_min": (metric, "min"),
            f"{metric}_max": (metric, "max"),
            "fit_seconds_mean": ("fit_seconds", "mean"),
            "n_folds": (metric, "size"),
            "n_failed": ("_failed", "sum"),
        }
    )
    return summary.sort_values(f"{metric}_mean", ascending=False).reset_index(drop=True)


def compare_to_baseline(results, baseline, metric="accuracy", alpha=0.05):
    """Compare each variant to ``baseline`` on matched folds."""
    frame = to_frame(results).dropna(subset=[metric])
    if frame.empty or baseline not in set(frame["variant"]):
        present = sorted(set(frame["variant"])) if not frame.empty else []
        raise ValueError(f"baseline {baseline!r} not found; present: {present}")

    keys = ["dataset", "seed", "fold"]
    base = frame[frame["variant"] == baseline].set_index(keys)[metric]
    rows = []
    for name in sorted(set(frame["variant"]) - {baseline}):
        other = frame[frame["variant"] == name].set_index(keys)[metric]
        paired = pd.concat(
            [base.rename("base"), other.rename("other")], axis=1, join="inner"
        ).dropna()
        if paired.empty:
            continue
        delta = float(paired["other"].mean() - paired["base"].mean())
        if np.allclose(paired["other"], paired["base"]):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(paired["other"], paired["base"]).pvalue)
        significant = bool(p_value < alpha)
        verdict = "no difference" if not significant else ("better" if delta > 0 else "worse")
        rows.append({
            "variant": name, "n_pairs": len(paired),
            "baseline_mean": float(paired["base"].mean()),
            "variant_mean": float(paired["other"].mean()), "delta": delta,
            "p_value": p_value, "significant": significant, "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def to_markdown(frame, float_format="%.4f"):
    """Render a pandas table for experiment reports."""
    return frame.to_markdown(index=False, floatfmt=float_format.replace("%", ""))
