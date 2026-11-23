from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TOLERANCE = 0.005

EXPECTED = {
    "MEET valence_std (paired mean diff, t, d, CI)": (0.071, 6.15, 0.571, 0.375, 0.768),
    "MEET arousal_std (paired mean diff, t, d, CI)": (0.096, 7.20, 0.668, 0.467, 0.869),
    "MEET discrete_entropy (paired mean diff, t, d, CI)": (
        0.075,
        4.77,
        0.443,
        0.252,
        0.634,
    ),
    "MERP mean_valence_abs_dev (mean diff, t, d, CI)": (
        -0.061,
        -3.73,
        -0.465,
        -0.712,
        -0.218,
    ),
    "MERP mean_arousal_abs_dev (mean diff, t, d, CI)": (
        -0.063,
        -3.99,
        -0.497,
        -0.744,
        -0.249,
    ),
    "MEET OLS R2 (valence, arousal)": (0.025, 0.008),
    "MERP OLS R2 (valence, arousal)": (0.061, 0.070),
}


def paired_stats(
    wide: pd.DataFrame, metric: str
) -> tuple[float, float, float, float, float]:
    musician = wide[(metric, "musician")]
    non_musician = wide[(metric, "non_musician")]
    diff = musician - non_musician
    test = stats.ttest_rel(musician, non_musician)
    d = diff.mean() / diff.std(ddof=1)
    se = np.sqrt(1.0 / len(diff) + d**2 / (2.0 * len(diff)))
    return diff.mean(), test.statistic, d, d - 1.96 * se, d + 1.96 * se


def independent_stats(
    table: pd.DataFrame, outcome: str
) -> tuple[float, float, float, float, float]:
    trained = table.loc[table["training"] == "Yes", outcome]
    untrained = table.loc[table["training"] == "No", outcome]
    test = stats.ttest_ind(trained, untrained, equal_var=False)
    n_a, n_b = len(trained), len(untrained)
    pooled_sd = np.sqrt(
        ((n_a - 1) * trained.var(ddof=1) + (n_b - 1) * untrained.var(ddof=1))
        / (n_a + n_b - 2)
    )
    d = (trained.mean() - untrained.mean()) / pooled_sd
    se = np.sqrt(1.0 / n_a + 1.0 / n_b + d**2 / (2.0 * (n_a + n_b)))
    return (
        trained.mean() - untrained.mean(),
        test.statistic,
        d,
        d - 1.96 * se,
        d + 1.96 * se,
    )


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    meet_group = pd.read_csv(processed / "meet_stimulus_group_disagreement_summary.csv")
    meet_participants = pd.read_csv(
        processed / "meet_listener_factor_analysis_table.csv"
    )
    merp_workers = pd.read_csv(processed / "merp_listener_validation_table.csv")

    wide = meet_group.pivot(
        index="stimulus_id",
        columns="group_label",
        values=["valence_std", "arousal_std", "discrete_entropy"],
    )

    computed = {}
    for metric in ("valence_std", "arousal_std", "discrete_entropy"):
        computed[f"MEET {metric} (paired mean diff, t, d, CI)"] = paired_stats(
            wide, metric
        )
    for outcome in ("mean_valence_abs_dev", "mean_arousal_abs_dev"):
        computed[f"MERP {outcome} (mean diff, t, d, CI)"] = independent_stats(
            merp_workers, outcome
        )
    computed["MEET OLS R2 (valence, arousal)"] = tuple(
        smf.ols(f"{outcome} ~ group_bin + age", data=meet_participants).fit().rsquared
        for outcome in ("mean_valence_abs_dev", "mean_arousal_abs_dev")
    )
    computed["MERP OLS R2 (valence, arousal)"] = tuple(
        smf.ols(f"{outcome} ~ training_bin + age_num", data=merp_workers).fit().rsquared
        for outcome in ("mean_valence_abs_dev", "mean_arousal_abs_dev")
    )

    failures = []
    for label, expected in EXPECTED.items():
        values = computed[label]
        print(f"{label}")
        print(f"  recomputed: {tuple(round(float(v), 4) for v in values)}")
        print(f"  reported:   {expected}")
        for value, target in zip(values, expected):
            if abs(float(value) - target) > TOLERANCE:
                failures.append(f"{label}: {value:.4f} != {target}")

    if failures:
        print("\nFAILED checks:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nAll recomputed statistics match the reported values.")


if __name__ == "__main__":
    main()
