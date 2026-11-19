from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import PROCESSED_DIR, TABLES_DIR

Z_CRITICAL_95 = 1.96

MEET_STIMULUS_OUTCOMES = ("valence_std", "arousal_std", "discrete_entropy")
MERP_WORKER_OUTCOMES = ("mean_valence_abs_dev", "mean_arousal_abs_dev")


def paired_d(diff: pd.Series) -> float:
    return float(diff.mean() / diff.std(ddof=1))


def independent_d(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    n_a, n_b = len(a), len(b)
    pooled_sd = float(
        np.sqrt(
            ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
        )
    )
    return float(a.mean() - b.mean()) / pooled_sd, pooled_sd


def paired_d_ci(d: float, n: int) -> tuple[float, float]:
    se = float(np.sqrt(1.0 / n + d**2 / (2.0 * n)))
    return d - Z_CRITICAL_95 * se, d + Z_CRITICAL_95 * se


def independent_d_ci(d: float, n_a: int, n_b: int) -> tuple[float, float]:
    se = float(np.sqrt(1.0 / n_a + 1.0 / n_b + d**2 / (2.0 * (n_a + n_b))))
    return d - Z_CRITICAL_95 * se, d + Z_CRITICAL_95 * se


def build_effect_size_table() -> pd.DataFrame:
    meet_group = pd.read_csv(
        PROCESSED_DIR / "meet_stimulus_group_disagreement_summary.csv"
    )
    merp_workers = pd.read_csv(PROCESSED_DIR / "merp_listener_validation_table.csv")

    rows: list[dict[str, object]] = []

    wide = meet_group.pivot(
        index="stimulus_id",
        columns="group_label",
        values=list(MEET_STIMULUS_OUTCOMES),
    )
    for outcome in MEET_STIMULUS_OUTCOMES:
        musician = wide[(outcome, "musician")]
        non_musician = wide[(outcome, "non_musician")]
        diff = musician - non_musician
        test = stats.ttest_rel(musician, non_musician)
        d = paired_d(diff)
        ci_low, ci_high = paired_d_ci(d, len(diff))
        rows.append(
            {
                "dataset": "MEET",
                "analysis_level": "stimulus_level",
                "outcome": outcome,
                "comparison": "musician_minus_non_musician",
                "mean_difference": float(diff.mean()),
                "statistic": float(test.statistic),
                "df": len(diff) - 1,
                "p_value": float(test.pvalue),
                "cohen_d": d,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n": int(len(diff)),
            }
        )

    for outcome in MERP_WORKER_OUTCOMES:
        trained = merp_workers.loc[merp_workers["training"] == "Yes", outcome]
        untrained = merp_workers.loc[merp_workers["training"] == "No", outcome]
        test = stats.ttest_ind(trained, untrained, equal_var=False)
        diff = float(trained.mean() - untrained.mean())
        d, _ = independent_d(trained, untrained)
        ci_low, ci_high = independent_d_ci(d, len(trained), len(untrained))
        rows.append(
            {
                "dataset": "MERP",
                "analysis_level": "worker_level",
                "outcome": outcome,
                "comparison": "trained_minus_untrained",
                "mean_difference": diff,
                "statistic": float(test.statistic),
                "df": float(test.df),
                "p_value": float(test.pvalue),
                "cohen_d": d,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n": int(len(trained) + len(untrained)),
            }
        )

    return pd.DataFrame(rows)


def write_effect_size_table(output_path: Path | None = None) -> pd.DataFrame:
    table = build_effect_size_table()
    destination = output_path or (TABLES_DIR / "effect_sizes_primary_comparisons.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


if __name__ == "__main__":
    result = write_effect_size_table()
    print(result.round(4).to_string(index=False))
