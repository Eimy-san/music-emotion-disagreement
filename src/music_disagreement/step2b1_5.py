from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CHECKPOINTS_DIR, TABLES_DIR


@dataclass
class Step2B15Artifacts:
    result_tier_table: pd.DataFrame
    variable_freeze_table: pd.DataFrame
    note_path: Path


def _ensure_parent_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meet = pd.read_csv(TABLES_DIR / "step2b1_meet_core_results.csv")
    merp = pd.read_csv(TABLES_DIR / "step2b1_merp_validation_results.csv")
    metric = pd.read_csv(TABLES_DIR / "step2a_metric_summary.csv")
    retention = pd.read_csv(TABLES_DIR / "step2a_sample_retention.csv")
    return meet, merp, metric, retention


def _pick(df: pd.DataFrame, section: str, outcome: str, predictor: str) -> pd.Series:
    row = df[
        (df["section"] == section)
        & (df["outcome"] == outcome)
        & (df["predictor"] == predictor)
    ]
    return row.iloc[0]


def build_result_tier_table(meet: pd.DataFrame, merp: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "dataset": "MEET",
            "result_id": "meet_stimulus_group_valence_std",
            "outcome": "stimulus-level valence_std",
            "predictor": "musician vs non_musician",
            "tier": "robust",
            "status_for_writing": "main_text",
            "summary": "Musician group shows higher stimulus-level valence disagreement than non-musician group.",
            "evidence": "paired_mean_difference=0.071; p=1.14e-08",
            "reason": "Stable direction, strong p-value, and aligned with the frozen stimulus-level MEET analysis.",
        },
        {
            "dataset": "MEET",
            "result_id": "meet_stimulus_group_arousal_std",
            "outcome": "stimulus-level arousal_std",
            "predictor": "musician vs non_musician",
            "tier": "robust",
            "status_for_writing": "main_text",
            "summary": "Musician group shows higher stimulus-level arousal disagreement than non-musician group.",
            "evidence": "paired_mean_difference=0.096; p=6.84e-11",
            "reason": "Strongest continuous MEET result and consistent with the same direction as valence_std.",
        },
        {
            "dataset": "MEET",
            "result_id": "meet_stimulus_group_discrete_entropy",
            "outcome": "stimulus-level discrete_entropy",
            "predictor": "musician vs non_musician",
            "tier": "robust",
            "status_for_writing": "supporting_text",
            "summary": "Musician group also shows higher discrete-label disagreement.",
            "evidence": "paired_mean_difference=0.075; p=5.36e-06",
            "reason": "Useful cross-check, but secondary because the project's cross-dataset bridge is continuous disagreement.",
        },
        {
            "dataset": "MEET",
            "result_id": "meet_participant_age_valence_dev",
            "outcome": "participant-level mean_valence_abs_dev",
            "predictor": "age",
            "tier": "directional_but_weak",
            "status_for_writing": "supporting_text",
            "summary": "Older participants show slightly lower valence deviation from consensus.",
            "evidence": "r=-0.145; p=0.040; model_a age p=0.049",
            "reason": "Small effect, only on one participant-level outcome, and not mirrored strongly elsewhere.",
        },
        {
            "dataset": "MEET",
            "result_id": "meet_participant_group_dev",
            "outcome": "participant-level mean_valence_abs_dev / mean_arousal_abs_dev",
            "predictor": "musician vs non_musician",
            "tier": "exploratory_not_for_main_text",
            "status_for_writing": "appendix_or_note",
            "summary": "Participant-level group differences are in the same direction as stimulus-level results but are not statistically convincing.",
            "evidence": "valence p=0.352; arousal p=0.220",
            "reason": "Directionally suggestive only; not strong enough for main written claims.",
        },
        {
            "dataset": "MEET",
            "result_id": "meet_music_background_continuous_vars",
            "outcome": "participant-level disagreement outcomes",
            "predictor": "years_musical_education / musical_experience / hours_musical_listening / bai / bdi",
            "tier": "exploratory_not_for_main_text",
            "status_for_writing": "appendix_or_note",
            "summary": "Most continuous listener-factor associations are weak or near zero in the first-round models.",
            "evidence": "Most p-values > 0.10; model R^2 around 0.02-0.03",
            "reason": "Too unstable and diffuse for a main-text interpretation.",
        },
        {
            "dataset": "MERP",
            "result_id": "merp_training_valence_arousal_dev",
            "outcome": "worker-level mean_valence_abs_dev / mean_arousal_abs_dev",
            "predictor": "training Yes vs No",
            "tier": "robust",
            "status_for_writing": "main_text",
            "summary": "Training is associated with lower worker-level deviation from consensus in MERP.",
            "evidence": "valence diff=-0.061, p=2.39e-04; arousal diff=-0.063, p=8.80e-05",
            "reason": "Consistent across both continuous outcomes and survives the simple training+age OLS.",
        },
        {
            "dataset": "MERP",
            "result_id": "merp_age_dev",
            "outcome": "worker-level mean_valence_abs_dev / mean_arousal_abs_dev",
            "predictor": "age",
            "tier": "directional_but_weak",
            "status_for_writing": "supporting_text",
            "summary": "Older MERP workers tend to deviate slightly more from song-level consensus.",
            "evidence": "valence r=0.175, p=0.0048; arousal r=0.191, p=0.0020; OLS age terms weaker",
            "reason": "Direction exists, but interpretation is modest and not fully stable after adjustment.",
        },
        {
            "dataset": "CROSS_DATASET",
            "result_id": "cross_dataset_listener_claim",
            "outcome": "overall listener-factor story",
            "predictor": "MEET + MERP together",
            "tier": "directional_but_weak",
            "status_for_writing": "main_text_with_caution",
            "summary": "Listener-related variables matter, but the specific direction is not cleanly replicated across datasets.",
            "evidence": "MEET musician effect is positive at stimulus level; MERP training effect is negative at worker-level deviation",
            "reason": "Supports a cautious listener-factor claim, not a strong explanatory claim.",
        },
    ]
    return pd.DataFrame(rows)


def build_variable_freeze_table() -> pd.DataFrame:
    rows = [
        {
            "dataset": "MEET",
            "item_type": "participant_level_outcome",
            "name": "mean_valence_abs_dev",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Best participant-level bridge to listener-factor analysis and aligned with the continuous disagreement framing.",
        },
        {
            "dataset": "MEET",
            "item_type": "participant_level_outcome",
            "name": "mean_arousal_abs_dev",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Co-primary participant-level continuous outcome paired with valence.",
        },
        {
            "dataset": "MEET",
            "item_type": "stimulus_level_outcome",
            "name": "valence_std",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Frozen continuous stimulus-level disagreement outcome from Step 2A.",
        },
        {
            "dataset": "MEET",
            "item_type": "stimulus_level_outcome",
            "name": "arousal_std",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Frozen continuous stimulus-level disagreement outcome from Step 2A.",
        },
        {
            "dataset": "MEET",
            "item_type": "outcome",
            "name": "modal_label_disagree_rate",
            "freeze_status": "auxiliary",
            "writing_role": "supporting_text",
            "reason": "Useful participant-level discrete check, but not the cross-dataset main outcome.",
        },
        {
            "dataset": "MEET",
            "item_type": "outcome",
            "name": "discrete_entropy",
            "freeze_status": "auxiliary",
            "writing_role": "supporting_text",
            "reason": "Good robustness check but secondary to continuous disagreement.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "group",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Most interpretable and strongest MEET signal; keep separate from years_musical_education and musical_experience in multivariable models because of collinearity/overlap.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "age",
            "freeze_status": "primary",
            "writing_role": "supporting_text",
            "reason": "Shows the clearest participant-level association, though small.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "years_musical_education",
            "freeze_status": "exploratory",
            "writing_role": "supporting_text",
            "reason": "Conceptually relevant but weak in Step 2B1 and partly overlapping with group/musical_experience.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "musical_experience",
            "freeze_status": "exploratory",
            "writing_role": "supporting_text",
            "reason": "Conceptually relevant but weak in Step 2B1 and partly overlapping with group/years_musical_education.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "hours_musical_listening",
            "freeze_status": "exploratory",
            "writing_role": "supporting_text",
            "reason": "Retained as a light control, but no clear Step 2B1 signal.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "bai",
            "freeze_status": "exploratory",
            "writing_role": "not_for_main_text",
            "reason": "Weak and unstable in Step 2B1; keep out of main written claims.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "bdi",
            "freeze_status": "exploratory",
            "writing_role": "not_for_main_text",
            "reason": "Weak and unstable in Step 2B1; keep out of main written claims.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "hours_musical_study",
            "freeze_status": "excluded",
            "writing_role": "not_for_main_text",
            "reason": "Excluded because of overlap with education/training intensity and collinearity risk.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "years_of_study",
            "freeze_status": "excluded",
            "writing_role": "not_for_main_text",
            "reason": "Excluded because it is not music-specific enough for the core claim.",
        },
        {
            "dataset": "MEET",
            "item_type": "listener_factor",
            "name": "musical_listening",
            "freeze_status": "excluded",
            "writing_role": "not_for_main_text",
            "reason": "Excluded because it overlaps semantically with hours_musical_listening and is less interpretable.",
        },
        {
            "dataset": "MERP",
            "item_type": "participant_level_outcome",
            "name": "mean_valence_abs_dev",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Primary worker-level continuous validation outcome.",
        },
        {
            "dataset": "MERP",
            "item_type": "participant_level_outcome",
            "name": "mean_arousal_abs_dev",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Co-primary worker-level continuous validation outcome.",
        },
        {
            "dataset": "MERP",
            "item_type": "listener_factor",
            "name": "training",
            "freeze_status": "primary",
            "writing_role": "main_text",
            "reason": "Most stable and interpretable MERP listener variable in Step 2B1.",
        },
        {
            "dataset": "MERP",
            "item_type": "listener_factor",
            "name": "age",
            "freeze_status": "exploratory",
            "writing_role": "supporting_text",
            "reason": "Shows some direction but should be interpreted cautiously.",
        },
        {
            "dataset": "MERP",
            "item_type": "listener_factor",
            "name": "training_duration",
            "freeze_status": "excluded",
            "writing_role": "not_for_main_text",
            "reason": "Excluded because of dirty coding and poor stability.",
        },
        {
            "dataset": "MERP",
            "item_type": "listener_factor",
            "name": "gender",
            "freeze_status": "excluded",
            "writing_role": "not_for_main_text",
            "reason": "Deferred to keep the minimal validation focused and interpretable.",
        },
    ]
    return pd.DataFrame(rows)


def write_decision_note(
    meet: pd.DataFrame,
    merp: pd.DataFrame,
    metric: pd.DataFrame,
    retention: pd.DataFrame,
    result_tier_table: pd.DataFrame,
    variable_freeze_table: pd.DataFrame,
) -> Path:
    meet_stim_val = _pick(meet, "stimulus_level_group_comparison", "valence_std", "musician_minus_non_musician")
    meet_stim_aro = _pick(meet, "stimulus_level_group_comparison", "arousal_std", "musician_minus_non_musician")
    meet_group_val = _pick(meet, "participant_level_group_comparison", "mean_valence_abs_dev", "musician_minus_non_musician")
    meet_group_aro = _pick(meet, "participant_level_group_comparison", "mean_arousal_abs_dev", "musician_minus_non_musician")
    meet_age_val = _pick(meet, "participant_level_correlation", "mean_valence_abs_dev", "age")
    merp_train_val = _pick(merp, "worker_level_group_comparison", "mean_valence_abs_dev", "Yes_minus_No")
    merp_train_aro = _pick(merp, "worker_level_group_comparison", "mean_arousal_abs_dev", "Yes_minus_No")
    merp_age_val = _pick(merp, "worker_level_correlation", "mean_valence_abs_dev", "age_num")

    path = CHECKPOINTS_DIR / "step2b1_5_decision_note.md"
    text = f"""# Step 2B1.5 Decision Note

Date: 2026-03-22

## Conclusion

Step 2B1.5 completed.

The current results are good enough to freeze a writing-oriented main line, but they do **not** support a strong claim that listener factors explain disagreement on their own.
The safest interpretation is that listener-related variables are associated with disagreement in some settings, while substantial residual disagreement remains unexplained and should motivate Step 2B2.

## Outcome Freeze

### Conclusion

The main outcomes are now frozen.

- Participant-level primary outcomes:
  - `MEET`: `mean_valence_abs_dev`, `mean_arousal_abs_dev`
  - `MERP`: `mean_valence_abs_dev`, `mean_arousal_abs_dev`
- Stimulus-level primary outcomes:
  - `MEET`: `valence_std`, `arousal_std`
- Auxiliary / robustness outcomes:
  - `MEET`: `discrete_entropy`, `modal_label_disagree_rate`

Writing rule:

- Main text should prioritize continuous disagreement outcomes.
- Discrete outcomes should only be used as supporting consistency checks.

## Variable Freeze

### Conclusion

The listener-factor set is now divided into primary, exploratory, and excluded variables.

- `MEET` primary variables: `group`, `age`
- `MEET` exploratory variables: `years_musical_education`, `musical_experience`, `hours_musical_listening`, `bai`, `bdi`
- `MEET` excluded from main text: `hours_musical_study`, `years_of_study`, `musical_listening`
- `MERP` primary variable: `training`
- `MERP` exploratory variable: `age`
- `MERP` excluded from main text: `training_duration`, `gender`

Reasoning:

- Excluded variables were dropped because of collinearity risk, semantic overlap, instability, or weak interpretability.
- `group` and `training` are the clearest dataset-specific listener variables.
- `age` is kept because it shows some signal, but it is not strong enough to dominate the story.

## Result Tiering

### Conclusion

The current result set splits cleanly into three tiers.

1. Robust results
- `MEET` stimulus-level musician vs non-musician differences in `valence_std` and `arousal_std`
- `MEET` stimulus-level `discrete_entropy` group difference as a supporting robustness result
- `MERP` training differences in `mean_valence_abs_dev` and `mean_arousal_abs_dev`

2. Directional but weak results
- `MEET` age vs `mean_valence_abs_dev` (small negative association)
- `MERP` age vs worker-level deviation outcomes
- Cross-dataset claim that listener-related variables matter, but not in a single clean direction

3. Not recommended for main-text results
- `MEET` participant-level musician vs non-musician deviation comparisons
- `MEET` years of musical education, musical experience, hours of musical listening, `BAI`, and `BDI` as explanatory claims

## Writing-Line Decision

### Conclusion

The best current paper line is cautious and asymmetrical.

1. Best current main line:
- In `MEET`, disagreement is systematically higher for stimuli as judged by the musician group than by the non-musician group.
- In `MERP`, a listener-related factor (`training`) is also associated with disagreement, but in a worker-level and not directly same-direction way.
- Together these findings suggest that disagreement is partly listener-related, but not yet sufficiently explained by listener factors alone.

2. Is the strong claim \"listener factors explain disagreement\" justified now?
- No.

3. Safer wording:
- \"Listener factors are associated with disagreement, but the current evidence suggests only partial and dataset-dependent explanatory value.\"
- \"Listener-related variables account for some variation in disagreement, while substantial residual disagreement remains.\"

4. Is Step 2B2 worth doing?
- Yes.

5. What role should Step 2B2 play?
- Primary role: explain residual disagreement
- Secondary role: refine or contextualize the mixed listener-factor pattern
- Not the right framing: simply \"confirm the listener-factor story\"

## Anchoring Facts From Step 2A and Step 2B1

### Conclusion

The main writing decisions remain grounded in the existing retained sample and frozen metric.

- Frozen metric: `std` for continuous disagreement
- `MEET` retained analysis items: {int(metric.loc[metric['dataset'] == 'MEET', 'n_formal_items'].iloc[0])} stimuli
- `MERP` retained formal validation items: {int(metric.loc[metric['dataset'] == 'MERP', 'n_formal_items'].iloc[0])} songs
- `MEET` stimulus-level group signal:
  - `valence_std`: diff = {meet_stim_val['estimate']:.3f}, p = {meet_stim_val['p_value']:.3g}
  - `arousal_std`: diff = {meet_stim_aro['estimate']:.3f}, p = {meet_stim_aro['p_value']:.3g}
- `MEET` participant-level group signal:
  - `mean_valence_abs_dev`: diff = {meet_group_val['estimate']:.3f}, p = {meet_group_val['p_value']:.3g}
  - `mean_arousal_abs_dev`: diff = {meet_group_aro['estimate']:.3f}, p = {meet_group_aro['p_value']:.3g}
- `MEET` participant-level age signal:
  - `mean_valence_abs_dev`: r = {meet_age_val['estimate']:.3f}, p = {meet_age_val['p_value']:.3g}
- `MERP` training signal:
  - `mean_valence_abs_dev`: diff = {merp_train_val['estimate']:.3f}, p = {merp_train_val['p_value']:.3g}
  - `mean_arousal_abs_dev`: diff = {merp_train_aro['estimate']:.3f}, p = {merp_train_aro['p_value']:.3g}
- `MERP` age signal:
  - `mean_valence_abs_dev`: r = {merp_age_val['estimate']:.3f}, p = {merp_age_val['p_value']:.3g}

## Stop Point

### Conclusion

Step 2B1.5 is complete.

This stage freezes the writing-usable interpretation of the current listener-factor results and stops before any Step 2B2 feature work or audio analysis.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_step2b1_5() -> Step2B15Artifacts:
    _ensure_parent_dirs()
    meet, merp, metric, retention = _load_inputs()
    result_tier_table = build_result_tier_table(meet, merp)
    variable_freeze_table = build_variable_freeze_table()
    result_tier_table.to_csv(TABLES_DIR / "result_tier_table.csv", index=False)
    variable_freeze_table.to_csv(TABLES_DIR / "variable_freeze_table.csv", index=False)
    note_path = write_decision_note(
        meet=meet,
        merp=merp,
        metric=metric,
        retention=retention,
        result_tier_table=result_tier_table,
        variable_freeze_table=variable_freeze_table,
    )
    return Step2B15Artifacts(
        result_tier_table=result_tier_table,
        variable_freeze_table=variable_freeze_table,
        note_path=note_path,
    )


if __name__ == "__main__":
    artifacts = run_step2b1_5()
    print("Step 2B1.5 complete.")
    print(f"Result tier rows: {len(artifacts.result_tier_table)}")
    print(f"Variable freeze rows: {len(artifacts.variable_freeze_table)}")
    print(f"Decision note: {artifacts.note_path}")
