from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import entropy


def _clean_numeric(values: Iterable[float]) -> np.ndarray:
    arr = pd.Series(list(values), dtype="float64").dropna().to_numpy()
    return arr


def continuous_disagreement(values: Iterable[float], metric: str = "std") -> float:
    arr = _clean_numeric(values)
    if arr.size == 0:
        return float("nan")
    if metric == "std":
        return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if metric == "mad":
        center = float(np.median(arr))
        return float(np.median(np.abs(arr - center)))
    if metric == "iqr":
        return float(np.percentile(arr, 75) - np.percentile(arr, 25))
    raise ValueError(f"Unsupported continuous metric: {metric}")


def discrete_disagreement(values: Iterable[object], metric: str = "entropy") -> float:
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return float("nan")
    probs = series.value_counts(normalize=True, dropna=True).to_numpy()
    if metric == "entropy":
        return float(entropy(probs, base=2))
    if metric == "top_label_proportion":
        return float(probs.max())
    raise ValueError(f"Unsupported discrete metric: {metric}")


def build_disagreement_table(
    df: pd.DataFrame,
    group_col: str,
    rating_col: str,
    metric: str = "std",
    label_type: str = "continuous",
) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col):
        values = group_df[rating_col].tolist()
        if label_type == "continuous":
            score = continuous_disagreement(values, metric=metric)
        else:
            score = discrete_disagreement(values, metric=metric)
        rows.append(
            {
                group_col: group_value,
                "n_ratings": int(group_df[rating_col].notna().sum()),
                "disagreement_metric": metric,
                "disagreement_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("disagreement_score", ascending=False).reset_index(drop=True)

