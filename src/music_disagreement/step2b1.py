from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .config import CHECKPOINTS_DIR, PROCESSED_DIR, TABLES_DIR
from .step2a import load_meet_raw

MEET_INCLUDED_VARS = (
    "group",
    "age",
    "years_musical_education",
    "musical_experience",
    "hours_musical_listening",
    "bai",
    "bdi",
)
MEET_EXCLUDED_VARS: Dict[str, str] = {
    "hours_musical_study": "Overlaps with education/training intensity and is excluded to keep the first-pass models light.",
    "years_of_study": "General education variable, not music-specific enough for the Step 2B1 core story.",
    "musical_listening": "Partially redundant with hours_musical_listening and less directly interpretable.",
}
MERP_INCLUDED_VARS = ("training", "age")
MERP_EXCLUDED_VARS: Dict[str, str] = {
    "training_duration": "Dirty coding in Step 2A outputs ({}, 05, 1995, -1, etc.), so it is not stable enough for minimal validation.",
    "gender": "Deferred to keep Step 2B1 focused on the most defensible listener variables.",
}
MEET_OUTCOMES = (
    "mean_valence_abs_dev",
    "mean_arousal_abs_dev",
    "modal_label_disagree_rate",
)
MERP_OUTCOMES = ("mean_valence_abs_dev", "mean_arousal_abs_dev")


@dataclass
class Step2B1Artifacts:
    meet_analysis_table: pd.DataFrame
    merp_validation_table: pd.DataFrame
    meet_results: pd.DataFrame
    merp_results: pd.DataFrame
    note_path: Path


def _ensure_parent_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_pearsonr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = pd.DataFrame({"x": x_num, "y": y_num}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan"), float("nan")
    result = stats.pearsonr(valid["x"], valid["y"])
    return float(result.statistic), float(result.pvalue)


def _welch_group_row(
    df: pd.DataFrame,
    dataset: str,
    analysis: str,
    outcome: str,
    group_col: str,
    group_a: str,
    group_b: str,
) -> Dict[str, object]:
    a = df.loc[df[group_col] == group_a, outcome]
    b = df.loc[df[group_col] == group_b, outcome]
    test = stats.ttest_ind(a, b, equal_var=False)
    return {
        "dataset": dataset,
        "section": analysis,
        "outcome": outcome,
        "predictor": f"{group_a}_minus_{group_b}",
        "estimate": float(a.mean() - b.mean()),
        "estimate_label": "mean_difference",
        "group_a": group_a,
        "group_b": group_b,
        "group_a_mean": float(a.mean()),
        "group_b_mean": float(b.mean()),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "n": int(len(a) + len(b)),
        "extra": "welch_t_test",
    }


def _paired_rows_from_meet_group_summary(
    meet_group_summary: pd.DataFrame,
) -> List[Dict[str, object]]:
    wide = meet_group_summary.pivot(
        index="stimulus_id",
        columns="group_label",
        values=["valence_std", "arousal_std", "discrete_entropy"],
    )
    wide.columns = [f"{metric}_{group_label}" for metric, group_label in wide.columns]
    wide = wide.reset_index()
    rows: List[Dict[str, object]] = []
    for outcome in ("valence_std", "arousal_std", "discrete_entropy"):
        musician = wide[f"{outcome}_musician"]
        non_musician = wide[f"{outcome}_non_musician"]
        diff = musician - non_musician
        paired_t = stats.ttest_rel(musician, non_musician)
        wilcoxon = stats.wilcoxon(diff)
        rows.append(
            {
                "dataset": "MEET",
                "section": "stimulus_level_group_comparison",
                "outcome": outcome,
                "predictor": "musician_minus_non_musician",
                "estimate": float(diff.mean()),
                "estimate_label": "paired_mean_difference",
                "group_a": "musician",
                "group_b": "non_musician",
                "group_a_mean": float(musician.mean()),
                "group_b_mean": float(non_musician.mean()),
                "statistic": float(paired_t.statistic),
                "p_value": float(paired_t.pvalue),
                "n": int(len(wide)),
                "extra": f"paired_t_test;wilcoxon_p={wilcoxon.pvalue:.6g};positive_diffs={int((diff > 0).sum())}",
            }
        )
    return rows


def _correlation_rows(
    df: pd.DataFrame,
    dataset: str,
    analysis: str,
    predictors: Sequence[str],
    outcomes: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for outcome in outcomes:
        for predictor in predictors:
            r_value, p_value = _safe_pearsonr(df[predictor], df[outcome])
            rows.append(
                {
                    "dataset": dataset,
                    "section": analysis,
                    "outcome": outcome,
                    "predictor": predictor,
                    "estimate": r_value,
                    "estimate_label": "pearson_r",
                    "group_a": "",
                    "group_b": "",
                    "group_a_mean": float("nan"),
                    "group_b_mean": float("nan"),
                    "statistic": r_value,
                    "p_value": p_value,
                    "n": int(df[[predictor, outcome]].dropna().shape[0]),
                    "extra": "pearson_correlation",
                }
            )
    return rows


def _ols_rows(
    df: pd.DataFrame,
    dataset: str,
    outcome: str,
    model_name: str,
    formula: str,
    keep_predictors: Sequence[str],
) -> List[Dict[str, object]]:
    fitted = smf.ols(formula=formula, data=df).fit()
    rows: List[Dict[str, object]] = []
    for predictor in keep_predictors:
        rows.append(
            {
                "dataset": dataset,
                "section": "ols_model",
                "outcome": outcome,
                "predictor": predictor,
                "estimate": float(fitted.params[predictor]),
                "estimate_label": model_name,
                "group_a": "",
                "group_b": "",
                "group_a_mean": float("nan"),
                "group_b_mean": float("nan"),
                "statistic": float(fitted.tvalues[predictor]),
                "p_value": float(fitted.pvalues[predictor]),
                "n": int(fitted.nobs),
                "extra": f"model={model_name};r_squared={fitted.rsquared:.6f}",
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "section": "ols_model_fit",
            "outcome": outcome,
            "predictor": model_name,
            "estimate": float(fitted.rsquared),
            "estimate_label": "r_squared",
            "group_a": "",
            "group_b": "",
            "group_a_mean": float("nan"),
            "group_b_mean": float("nan"),
            "statistic": float(fitted.fvalue)
            if fitted.fvalue is not None
            else float("nan"),
            "p_value": float(fitted.f_pvalue)
            if fitted.f_pvalue is not None
            else float("nan"),
            "n": int(fitted.nobs),
            "extra": f"formula={formula}",
        }
    )
    return rows


def build_meet_listener_factor_analysis_table() -> pd.DataFrame:
    meet_long = pd.read_csv(PROCESSED_DIR / "meet_participant_stimulus_long.csv")
    meet_raw = load_meet_raw()
    consensus = (
        meet_long.groupby("stimulus_id")
        .agg(
            consensus_valence=("valence_rating", "mean"),
            consensus_arousal=("arousal_rating", "mean"),
        )
        .reset_index()
    )
    modal_labels = (
        meet_long.groupby("stimulus_id")["discrete_emotion_label"]
        .agg(lambda series: series.mode().iloc[0])
        .rename("modal_discrete_label")
        .reset_index()
    )
    enriched = meet_long.merge(consensus, on="stimulus_id", how="left").merge(
        modal_labels, on="stimulus_id", how="left"
    )
    enriched["valence_abs_dev"] = (
        enriched["valence_rating"] - enriched["consensus_valence"]
    ).abs()
    enriched["arousal_abs_dev"] = (
        enriched["arousal_rating"] - enriched["consensus_arousal"]
    ).abs()
    enriched["modal_label_disagree"] = (
        enriched["discrete_emotion_label"] != enriched["modal_discrete_label"]
    ).astype(int)

    participant_profile = meet_raw[
        [
            "Participants",
            "Group",
            "Age",
            "Years of musical education",
            "Musical Experience",
            "Hours of musical listening",
            "BAI",
            "BDI",
            "Hours of musical study",
            "Years of study",
            "Musical listening",
        ]
    ].copy()
    participant_profile.columns = [
        "participant_id",
        "group",
        "age",
        "years_musical_education",
        "musical_experience",
        "hours_musical_listening",
        "bai",
        "bdi",
        "hours_musical_study",
        "years_of_study",
        "musical_listening",
    ]

    analysis_table = (
        enriched.groupby("participant_id")
        .agg(
            group=("group", "first"),
            group_label=("group_label", "first"),
            age_from_long=("age", "first"),
            mean_valence_abs_dev=("valence_abs_dev", "mean"),
            mean_arousal_abs_dev=("arousal_abs_dev", "mean"),
            modal_label_disagree_rate=("modal_label_disagree", "mean"),
            n_stimuli=("stimulus_id", "size"),
        )
        .reset_index()
        .merge(participant_profile, on=["participant_id", "group"], how="left")
    )
    analysis_table["group_bin"] = (analysis_table["group_label"] == "musician").astype(
        int
    )
    return analysis_table.sort_values("participant_id").reset_index(drop=True)


def build_merp_listener_validation_table() -> pd.DataFrame:
    merp_trials = pd.read_csv(PROCESSED_DIR / "merp_approved_worker_song_trials.csv")
    main_trials = merp_trials[~merp_trials["is_deam_slot"]].copy()
    song_consensus = (
        main_trials.groupby("song_url")
        .agg(
            consensus_valence=("valence_mean", "mean"),
            consensus_arousal=("arousal_mean", "mean"),
        )
        .reset_index()
    )
    main_trials = main_trials.merge(song_consensus, on="song_url", how="left")
    main_trials["valence_abs_dev"] = (
        main_trials["valence_mean"] - main_trials["consensus_valence"]
    ).abs()
    main_trials["arousal_abs_dev"] = (
        main_trials["arousal_mean"] - main_trials["consensus_arousal"]
    ).abs()

    worker_table = (
        main_trials.groupby("worker_id")
        .agg(
            age=("age", "first"),
            training=("training", "first"),
            mean_valence_abs_dev=("valence_abs_dev", "mean"),
            mean_arousal_abs_dev=("arousal_abs_dev", "mean"),
            n_trials=("song_url", "size"),
        )
        .reset_index()
    )
    worker_table["age_num"] = pd.to_numeric(worker_table["age"], errors="coerce")
    worker_table["training_bin"] = worker_table["training"].map({"Yes": 1, "No": 0})
    return worker_table.sort_values("worker_id").reset_index(drop=True)


def build_meet_results(
    meet_table: pd.DataFrame,
    meet_group_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rows.extend(_paired_rows_from_meet_group_summary(meet_group_summary))
    for outcome in MEET_OUTCOMES:
        rows.append(
            _welch_group_row(
                meet_table,
                dataset="MEET",
                analysis="participant_level_group_comparison",
                outcome=outcome,
                group_col="group_label",
                group_a="musician",
                group_b="non_musician",
            )
        )

    rows.extend(
        _correlation_rows(
            meet_table,
            dataset="MEET",
            analysis="participant_level_correlation",
            predictors=(
                "age",
                "years_musical_education",
                "musical_experience",
                "hours_musical_listening",
                "bai",
                "bdi",
            ),
            outcomes=MEET_OUTCOMES,
        )
    )

    model_specs = {
        "model_group_age": (
            "{outcome} ~ group_bin + age",
            ("group_bin", "age"),
        ),
        "model_a": (
            "{outcome} ~ group_bin + age + hours_musical_listening + bai + bdi",
            ("group_bin", "age", "hours_musical_listening", "bai", "bdi"),
        ),
        "model_b": (
            "{outcome} ~ years_musical_education + age + hours_musical_listening + bai + bdi",
            ("years_musical_education", "age", "hours_musical_listening", "bai", "bdi"),
        ),
        "model_c": (
            "{outcome} ~ musical_experience + age + hours_musical_listening + bai + bdi",
            ("musical_experience", "age", "hours_musical_listening", "bai", "bdi"),
        ),
    }
    for outcome in ("mean_valence_abs_dev", "mean_arousal_abs_dev"):
        for model_name, (formula_template, predictors) in model_specs.items():
            rows.extend(
                _ols_rows(
                    meet_table,
                    dataset="MEET",
                    outcome=outcome,
                    model_name=model_name,
                    formula=formula_template.format(outcome=outcome),
                    keep_predictors=predictors,
                )
            )
    return pd.DataFrame(rows)


def build_merp_results(merp_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for outcome in MERP_OUTCOMES:
        rows.append(
            _welch_group_row(
                merp_table,
                dataset="MERP",
                analysis="worker_level_group_comparison",
                outcome=outcome,
                group_col="training",
                group_a="Yes",
                group_b="No",
            )
        )

    rows.extend(
        _correlation_rows(
            merp_table,
            dataset="MERP",
            analysis="worker_level_correlation",
            predictors=("age_num",),
            outcomes=MERP_OUTCOMES,
        )
    )

    for outcome in MERP_OUTCOMES:
        rows.extend(
            _ols_rows(
                merp_table,
                dataset="MERP",
                outcome=outcome,
                model_name="model_training_age",
                formula=f"{outcome} ~ training_bin + age_num",
                keep_predictors=("training_bin", "age_num"),
            )
        )
    return pd.DataFrame(rows)


def _get_result_row(
    results: pd.DataFrame, section: str, outcome: str, predictor: str
) -> pd.Series:
    row = results[
        (results["section"] == section)
        & (results["outcome"] == outcome)
        & (results["predictor"] == predictor)
    ]
    return row.iloc[0]


def _step2b2_recommendation(
    meet_results: pd.DataFrame, merp_results: pd.DataFrame
) -> tuple[str, str]:
    meet_stim_val = _get_result_row(
        meet_results,
        "stimulus_level_group_comparison",
        "valence_std",
        "musician_minus_non_musician",
    )
    meet_stim_aro = _get_result_row(
        meet_results,
        "stimulus_level_group_comparison",
        "arousal_std",
        "musician_minus_non_musician",
    )
    merp_train_val = _get_result_row(
        merp_results,
        "worker_level_group_comparison",
        "mean_valence_abs_dev",
        "Yes_minus_No",
    )
    merp_train_aro = _get_result_row(
        merp_results,
        "worker_level_group_comparison",
        "mean_arousal_abs_dev",
        "Yes_minus_No",
    )
    if (
        meet_stim_val["p_value"] < 0.05
        or meet_stim_aro["p_value"] < 0.05
        or merp_train_val["p_value"] < 0.05
        or merp_train_aro["p_value"] < 0.05
    ):
        return (
            "worth_entering_step2b2",
            "Step 2B2 is worth doing because Step 2B1 found at least one stable listener-related signal, even though the cross-dataset direction is mixed rather than cleanly replicated.",
        )
    return (
        "not_yet_worth_entering_step2b2",
        "Step 2B2 is not yet justified because Step 2B1 did not recover a stable listener-factor signal.",
    )


def write_step2b1_note(
    meet_table: pd.DataFrame,
    merp_table: pd.DataFrame,
    meet_results: pd.DataFrame,
    merp_results: pd.DataFrame,
) -> Path:
    meet_stim_val = _get_result_row(
        meet_results,
        "stimulus_level_group_comparison",
        "valence_std",
        "musician_minus_non_musician",
    )
    meet_stim_aro = _get_result_row(
        meet_results,
        "stimulus_level_group_comparison",
        "arousal_std",
        "musician_minus_non_musician",
    )
    meet_stim_disc = _get_result_row(
        meet_results,
        "stimulus_level_group_comparison",
        "discrete_entropy",
        "musician_minus_non_musician",
    )
    meet_group_val = _get_result_row(
        meet_results,
        "participant_level_group_comparison",
        "mean_valence_abs_dev",
        "musician_minus_non_musician",
    )
    meet_group_aro = _get_result_row(
        meet_results,
        "participant_level_group_comparison",
        "mean_arousal_abs_dev",
        "musician_minus_non_musician",
    )
    meet_age_val = _get_result_row(
        meet_results, "participant_level_correlation", "mean_valence_abs_dev", "age"
    )
    merp_train_val = _get_result_row(
        merp_results,
        "worker_level_group_comparison",
        "mean_valence_abs_dev",
        "Yes_minus_No",
    )
    merp_train_aro = _get_result_row(
        merp_results,
        "worker_level_group_comparison",
        "mean_arousal_abs_dev",
        "Yes_minus_No",
    )
    merp_age_val = _get_result_row(
        merp_results, "worker_level_correlation", "mean_valence_abs_dev", "age_num"
    )
    recommendation_code, recommendation_text = _step2b2_recommendation(
        meet_results, merp_results
    )

    path = CHECKPOINTS_DIR / "step2b1_analysis_note.md"
    note = f"""# Step 2B1 Analysis Note

Date: 2026-03-22

## Conclusion

Step 2B1 completed.

The first-round listener-factor analysis gives a mixed result rather than a clean single-direction story.
`MEET` shows a stable stimulus-level musician vs non-musician disagreement difference, but the participant-level listener-factor associations are mostly weak.
`MERP` gives a minimal external signal for listener-related variables, but its direction is not a clean replication of the `MEET` musician effect.

## Variables Analyzed

### Conclusion

The final Step 2B1 analysis used a deliberately small, interpretable variable set.

- `MEET` included: {", ".join(MEET_INCLUDED_VARS)}
- `MERP` included: {", ".join(MERP_INCLUDED_VARS)}

Variables excluded in this round:

{chr(10).join(f"- `MEET` {name}: {reason}" for name, reason in MEET_EXCLUDED_VARS.items())}
{chr(10).join(f"- `MERP` {name}: {reason}" for name, reason in MERP_EXCLUDED_VARS.items())}

## Main Findings

### Conclusion

The strongest `MEET` signal is group-based and appears at the stimulus level, not the participant-level listener-factor models.

- `MEET` stimulus-level paired comparison:
  - `valence_std`: musician minus non-musician mean difference = {meet_stim_val["estimate"]:.3f}, p = {meet_stim_val["p_value"]:.3g}
  - `arousal_std`: musician minus non-musician mean difference = {meet_stim_aro["estimate"]:.3f}, p = {meet_stim_aro["p_value"]:.3g}
  - `discrete_entropy`: musician minus non-musician mean difference = {meet_stim_disc["estimate"]:.3f}, p = {meet_stim_disc["p_value"]:.3g}
- `MEET` participant-level group comparison:
  - `mean_valence_abs_dev`: musician minus non-musician = {meet_group_val["estimate"]:.3f}, p = {meet_group_val["p_value"]:.3g}
  - `mean_arousal_abs_dev`: musician minus non-musician = {meet_group_aro["estimate"]:.3f}, p = {meet_group_aro["p_value"]:.3g}
- `MEET` continuous listener factors:
  - Most correlations and OLS effects are weak.
  - The clearest single-variable signal is a small negative age correlation with `mean_valence_abs_dev` (r = {meet_age_val["estimate"]:.3f}, p = {meet_age_val["p_value"]:.3g}).

## Stability Assessment

### Conclusion

Only part of the result set looks stable enough to carry forward as a serious lead.

- More stable:
  - `MEET` stimulus-level musician vs non-musician disagreement difference
  - `MERP` training contrast for both continuous disagreement outcomes
- More exploratory:
  - `MEET` participant-level group effect
  - `MEET` years of musical education, musical experience, hours of musical listening, `BAI`, and `BDI`
  - `MERP` age trend

## MERP Validation

### Conclusion

`MERP` provides partial listener-related support, but not a clean directional replication of the `MEET` musician effect.

- `MERP` training comparison:
  - `mean_valence_abs_dev`: Yes minus No = {merp_train_val["estimate"]:.3f}, p = {merp_train_val["p_value"]:.3g}
  - `mean_arousal_abs_dev`: Yes minus No = {merp_train_aro["estimate"]:.3f}, p = {merp_train_aro["p_value"]:.3g}
- `MERP` age correlation with `mean_valence_abs_dev`: r = {merp_age_val["estimate"]:.3f}, p = {merp_age_val["p_value"]:.3g}

Interpretation:

- `MEET` suggests higher disagreement in the musician group at the stimulus level.
- `MERP` suggests lower participant-level disagreement among listeners with training.
- Therefore `MERP` supports the broader claim that listener-related variables matter, but it does not cleanly support the specific `MEET` musician-direction result.

## Step 2B2 Decision

### Conclusion

{recommendation_text}

Decision code: `{recommendation_code}`

Reason:

- Listener factors alone do not yet give a clean, uniform explanatory story.
- The mixed pattern leaves room for `cue-conflict` to add explanatory value in Step 2B2.
- Step 2B2 should therefore be framed as an attempt to explain residual disagreement, not as a guaranteed confirmation of the Step 2B1 direction.
"""
    path.write_text(note, encoding="utf-8")
    return path


def run_step2b1() -> Step2B1Artifacts:
    _ensure_parent_dirs()
    meet_group_summary = pd.read_csv(
        PROCESSED_DIR / "meet_stimulus_group_disagreement_summary.csv"
    )
    meet_analysis_table = build_meet_listener_factor_analysis_table()
    merp_validation_table = build_merp_listener_validation_table()
    meet_results = build_meet_results(meet_analysis_table, meet_group_summary)
    merp_results = build_merp_results(merp_validation_table)

    meet_analysis_table.to_csv(
        PROCESSED_DIR / "meet_listener_factor_analysis_table.csv", index=False
    )
    merp_validation_table.to_csv(
        PROCESSED_DIR / "merp_listener_validation_table.csv", index=False
    )
    meet_results.to_csv(TABLES_DIR / "step2b1_meet_core_results.csv", index=False)
    merp_results.to_csv(TABLES_DIR / "step2b1_merp_validation_results.csv", index=False)
    note_path = write_step2b1_note(
        meet_table=meet_analysis_table,
        merp_table=merp_validation_table,
        meet_results=meet_results,
        merp_results=merp_results,
    )
    return Step2B1Artifacts(
        meet_analysis_table=meet_analysis_table,
        merp_validation_table=merp_validation_table,
        meet_results=meet_results,
        merp_results=merp_results,
        note_path=note_path,
    )


if __name__ == "__main__":
    artifacts = run_step2b1()
    print("Step 2B1 complete.")
    print(f"MEET analysis rows: {len(artifacts.meet_analysis_table)}")
    print(f"MERP validation rows: {len(artifacts.merp_validation_table)}")
    print(f"MEET result rows: {len(artifacts.meet_results)}")
    print(f"MERP result rows: {len(artifacts.merp_results)}")
    print(f"Notes written to: {artifacts.note_path}")
