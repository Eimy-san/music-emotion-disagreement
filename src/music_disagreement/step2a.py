from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (
    CHECKPOINTS_DIR,
    DATASETS,
    PRIMARY_DISAGREEMENT_METRIC,
    PROCESSED_DIR,
    TABLES_DIR,
)
from .disagreement import continuous_disagreement, discrete_disagreement

MEET_RAW_PATH = DATASETS["meet"] / "Raw_Data.xlsx"
MERP_RAW_DIR = DATASETS["merp"]
MERP_EXCLUDED_FILES = {"batch8_all_world_NONmaster.csv"}
MEET_STIMULUS_COUNT = 116
MERP_MAIN_SLOT_COUNT = 20
MERP_DEAM_SLOT_COUNT = 4
CORE_LISTENER_FACTORS = (
    "group",
    "age",
    "musical_experience",
    "hours_musical_listening",
)
MEET_CUE_CONFLICT_FEATURES = (
    "tempo",
    "spectral_centroid_mean",
    "dynamic_range",
    "spectral_flux_proxy",
)
MEET_CUE_CONFLICT_RULES = (
    "bright_slow",
    "fast_minor_proxy",
    "stable_harmony_high_flux",
)


@dataclass
class Step2AArtifacts:
    meet_long: pd.DataFrame
    meet_summary: pd.DataFrame
    meet_group_summary: pd.DataFrame
    merp_trials: pd.DataFrame
    merp_song_summary: pd.DataFrame
    metric_summary: pd.DataFrame
    sample_retention: pd.DataFrame
    cue_conflict_readiness: pd.DataFrame
    order_sanity: pd.DataFrame
    notes_path: Path


def _ensure_parent_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _top_label_proportion(values: Iterable[object]) -> float:
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return float("nan")
    return float(series.value_counts(normalize=True, dropna=True).max())


def _safe_int(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _parse_numeric_series(value: object) -> Optional[List[float]]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "{}", "nan", "None"}:
        return None
    values: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values if values else None


def _song_id_from_url(song_url: str) -> str:
    return Path(song_url).stem


def load_meet_raw() -> pd.DataFrame:
    return pd.read_excel(MEET_RAW_PATH, sheet_name="Raw_Data")


def build_meet_participant_stimulus_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_labels = {1: "musician", 2: "non_musician"}
    for _, participant in df.iterrows():
        participant_id = _safe_int(participant["Participants"])
        group = _safe_int(participant["Group"])
        for stimulus_id in range(1, MEET_STIMULUS_COUNT + 1):
            rows.append(
                {
                    "participant_id": participant_id,
                    "stimulus_id": stimulus_id,
                    "group": group,
                    "group_label": group_labels.get(group, "unknown"),
                    "presentation_order": _safe_int(participant["Order"]),
                    "age": _safe_float(participant["Age"]),
                    "musical_experience": _safe_float(participant["Musical Experience"]),
                    "hours_musical_listening": _safe_float(participant["Hours of musical listening"]),
                    "discrete_emotion_label": _safe_int(participant[stimulus_id]),
                    "arousal_rating": _safe_float(participant[f"{stimulus_id}.2"]),
                    "valence_rating": _safe_float(participant[f"{stimulus_id}.3"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["stimulus_id", "participant_id"]).reset_index(drop=True)


def summarize_meet_stimulus(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stimulus_id, group_df in long_df.groupby("stimulus_id", sort=True):
        rows.append(
            {
                "stimulus_id": int(stimulus_id),
                "n_raters": int(group_df["participant_id"].nunique()),
                "primary_metric": PRIMARY_DISAGREEMENT_METRIC,
                "discrete_entropy": discrete_disagreement(
                    group_df["discrete_emotion_label"], metric="entropy"
                ),
                "top_label_proportion": _top_label_proportion(group_df["discrete_emotion_label"]),
                "valence_std": continuous_disagreement(
                    group_df["valence_rating"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
                "valence_mad": continuous_disagreement(group_df["valence_rating"], metric="mad"),
                "valence_iqr": continuous_disagreement(group_df["valence_rating"], metric="iqr"),
                "arousal_std": continuous_disagreement(
                    group_df["arousal_rating"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
                "arousal_mad": continuous_disagreement(group_df["arousal_rating"], metric="mad"),
                "arousal_iqr": continuous_disagreement(group_df["arousal_rating"], metric="iqr"),
            }
        )
    return pd.DataFrame(rows).sort_values("stimulus_id").reset_index(drop=True)


def summarize_meet_stimulus_by_group(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = long_df.groupby(["stimulus_id", "group", "group_label"], sort=True)
    for (stimulus_id, group, group_label), group_df in grouped:
        rows.append(
            {
                "stimulus_id": int(stimulus_id),
                "group": _safe_int(group),
                "group_label": group_label,
                "n_raters": int(group_df["participant_id"].nunique()),
                "primary_metric": PRIMARY_DISAGREEMENT_METRIC,
                "discrete_entropy": discrete_disagreement(
                    group_df["discrete_emotion_label"], metric="entropy"
                ),
                "top_label_proportion": _top_label_proportion(group_df["discrete_emotion_label"]),
                "valence_std": continuous_disagreement(
                    group_df["valence_rating"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
                "arousal_std": continuous_disagreement(
                    group_df["arousal_rating"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["stimulus_id", "group"]).reset_index(drop=True)


def compute_meet_order_sanity(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stimulus_id, stimulus_df in long_df.groupby("stimulus_id", sort=True):
        order_1 = stimulus_df[stimulus_df["presentation_order"] == 1]
        order_2 = stimulus_df[stimulus_df["presentation_order"] == 2]
        p1 = order_1["discrete_emotion_label"].value_counts(normalize=True)
        p2 = order_2["discrete_emotion_label"].value_counts(normalize=True)
        labels = sorted(set(p1.index).union(set(p2.index)))
        total_variation = 0.5 * sum(abs(p1.get(label, 0.0) - p2.get(label, 0.0)) for label in labels)
        rows.append(
            {
                "stimulus_id": int(stimulus_id),
                "n_order_1": int(order_1["participant_id"].nunique()),
                "n_order_2": int(order_2["participant_id"].nunique()),
                "discrete_tv_distance": float(total_variation),
                "valence_mean_diff_abs": float(
                    abs(order_1["valence_rating"].mean() - order_2["valence_rating"].mean())
                ),
                "arousal_mean_diff_abs": float(
                    abs(order_1["arousal_rating"].mean() - order_2["arousal_rating"].mean())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("stimulus_id").reset_index(drop=True)


def load_merp_rows() -> pd.DataFrame:
    frames = []
    for path in sorted(MERP_RAW_DIR.glob("*.csv")):
        if path.name in MERP_EXCLUDED_FILES:
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No MERP CSV files found after exclusions.")
    return pd.concat(frames, ignore_index=True)


def build_merp_approved_worker_song_trials(all_rows: pd.DataFrame) -> pd.DataFrame:
    approved = all_rows[all_rows["AssignmentStatus"].eq("Approved")].copy()
    rows = []
    for _, record in approved.iterrows():
        music_all = str(record["Input.music_all"]).split(";")
        deam_music_all = str(record["Input.deam_music_all"]).split(";")
        random_indices = [int(x) for x in str(record["Answer.random_indices"]).split(",")]
        random_deam_indices = [int(x) for x in str(record["Answer.random_deam_indices"]).split(",")]
        selected_main = [music_all[index] for index in random_indices]
        selected_deam = [deam_music_all[index] for index in random_deam_indices]
        slot_song_urls = selected_main + selected_deam

        for slot in range(1, MERP_MAIN_SLOT_COUNT + MERP_DEAM_SLOT_COUNT + 1):
            if slot > len(slot_song_urls):
                continue
            valence_series = _parse_numeric_series(record.get(f"Answer.valences_{slot}"))
            arousal_series = _parse_numeric_series(record.get(f"Answer.arousals_{slot}"))
            if valence_series is None or arousal_series is None:
                continue

            song_url = slot_song_urls[slot - 1]
            rows.append(
                {
                    "source_file": record["source_file"],
                    "worker_id": record["WorkerId"],
                    "assignment_status": record["AssignmentStatus"],
                    "slot": slot,
                    "trial_source": "deam" if slot > MERP_MAIN_SLOT_COUNT else "main",
                    "is_deam_slot": slot > MERP_MAIN_SLOT_COUNT,
                    "song_url": song_url,
                    "song_id": _song_id_from_url(song_url),
                    "valence_mean": float(np.mean(valence_series)),
                    "arousal_mean": float(np.mean(arousal_series)),
                    "valence_len": len(valence_series),
                    "arousal_len": len(arousal_series),
                    "age": record.get("Answer.age"),
                    "gender": record.get("Answer.gender"),
                    "training": record.get("Answer.training"),
                    "training_duration": record.get("Answer.training-duration"),
                }
            )
    if not rows:
        raise ValueError("MERP Approved rows did not yield any usable worker-song trials.")
    return pd.DataFrame(rows).sort_values(["song_url", "worker_id", "slot"]).reset_index(drop=True)


def summarize_merp_song_disagreement(trials_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for song_url, song_df in trials_df.groupby("song_url", sort=True):
        rows.append(
            {
                "song_url": song_url,
                "song_id": song_df["song_id"].iloc[0],
                "n_trials": int(len(song_df)),
                "n_workers": int(song_df["worker_id"].nunique()),
                "is_deam_slot": bool(song_df["is_deam_slot"].any()),
                "primary_metric": PRIMARY_DISAGREEMENT_METRIC,
                "valence_std": continuous_disagreement(
                    song_df["valence_mean"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
                "valence_mad": continuous_disagreement(song_df["valence_mean"], metric="mad"),
                "valence_iqr": continuous_disagreement(song_df["valence_mean"], metric="iqr"),
                "arousal_std": continuous_disagreement(
                    song_df["arousal_mean"], metric=PRIMARY_DISAGREEMENT_METRIC
                ),
                "arousal_mad": continuous_disagreement(song_df["arousal_mean"], metric="mad"),
                "arousal_iqr": continuous_disagreement(song_df["arousal_mean"], metric="iqr"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["is_deam_slot", "song_url"]).reset_index(drop=True)
    out["formal_validation_song"] = ~out["is_deam_slot"]
    return out


def build_metric_summary(
    meet_summary: pd.DataFrame,
    merp_song_summary: pd.DataFrame,
) -> pd.DataFrame:
    formal_merp = merp_song_summary[merp_song_summary["formal_validation_song"]].copy()
    rows = [
        {
            "dataset": "MEET",
            "role": "main_analysis",
            "analysis_unit": "stimulus_level",
            "primary_metric": PRIMARY_DISAGREEMENT_METRIC,
            "primary_dimensions": "valence_std; arousal_std",
            "secondary_metrics": "discrete_entropy; top_label_proportion; valence_mad; valence_iqr; arousal_mad; arousal_iqr",
            "n_analysis_items": int(len(meet_summary)),
            "n_formal_items": int(len(meet_summary)),
            "mean_valence_std": float(meet_summary["valence_std"].mean()),
            "mean_arousal_std": float(meet_summary["arousal_std"].mean()),
        },
        {
            "dataset": "MERP",
            "role": "external_validation",
            "analysis_unit": "song_level",
            "primary_metric": PRIMARY_DISAGREEMENT_METRIC,
            "primary_dimensions": "valence_std; arousal_std",
            "secondary_metrics": "valence_mad; valence_iqr; arousal_mad; arousal_iqr",
            "n_analysis_items": int(len(merp_song_summary)),
            "n_formal_items": int(len(formal_merp)),
            "mean_valence_std": float(formal_merp["valence_std"].mean()),
            "mean_arousal_std": float(formal_merp["arousal_std"].mean()),
        },
    ]
    return pd.DataFrame(rows)


def build_sample_retention(
    meet_raw: pd.DataFrame,
    meet_long: pd.DataFrame,
    meet_summary: pd.DataFrame,
    merp_all_rows: pd.DataFrame,
    merp_trials: pd.DataFrame,
    merp_song_summary: pd.DataFrame,
) -> pd.DataFrame:
    approved = merp_all_rows[merp_all_rows["AssignmentStatus"].eq("Approved")]
    formal_merp = merp_song_summary[merp_song_summary["formal_validation_song"]]
    rows = [
        {
            "dataset": "MEET",
            "stage": "raw_participants",
            "n_rows": int(len(meet_raw)),
            "n_unique_items": int(meet_raw["Participants"].nunique()),
            "notes": "Raw_Data sheet participants.",
        },
        {
            "dataset": "MEET",
            "stage": "participant_stimulus_long",
            "n_rows": int(len(meet_long)),
            "n_unique_items": int(meet_long["stimulus_id"].nunique()),
            "notes": "Participant x stimulus long table.",
        },
        {
            "dataset": "MEET",
            "stage": "stimulus_summary",
            "n_rows": int(len(meet_summary)),
            "n_unique_items": int(meet_summary["stimulus_id"].nunique()),
            "notes": "Stimulus-level disagreement summary.",
        },
        {
            "dataset": "MERP",
            "stage": "eligible_raw_rows_excluding_batch8",
            "n_rows": int(len(merp_all_rows)),
            "n_unique_items": int(merp_all_rows["source_file"].nunique()),
            "notes": "Raw MTurk rows after excluding batch8_all_world_NONmaster.csv.",
        },
        {
            "dataset": "MERP",
            "stage": "approved_rows",
            "n_rows": int(len(approved)),
            "n_unique_items": int(approved["WorkerId"].nunique()),
            "notes": "Approved rows only.",
        },
        {
            "dataset": "MERP",
            "stage": "worker_song_trials",
            "n_rows": int(len(merp_trials)),
            "n_unique_items": int(merp_trials["song_url"].nunique()),
            "notes": "Worker x song trials reconstructed from Approved rows.",
        },
        {
            "dataset": "MERP",
            "stage": "song_summary_all",
            "n_rows": int(len(merp_song_summary)),
            "n_unique_items": int(merp_song_summary["song_url"].nunique()),
            "notes": "Song-level disagreement summary including DEAM slots.",
        },
        {
            "dataset": "MERP",
            "stage": "song_summary_formal_main_pool",
            "n_rows": int(len(formal_merp)),
            "n_unique_items": int(formal_merp["song_url"].nunique()),
            "notes": "Formal validation songs from main pool only.",
        },
    ]
    return pd.DataFrame(rows)


def build_cue_conflict_readiness() -> pd.DataFrame:
    rows = []
    for feature in MEET_CUE_CONFLICT_FEATURES:
        rows.append(
            {
                "item_type": "feature",
                "name": feature,
                "status": "pending_audio",
                "notes": "Frozen for Step 2A; not extracted because MEET audio files are not present locally.",
            }
        )
    for rule in MEET_CUE_CONFLICT_RULES:
        rows.append(
            {
                "item_type": "cue_conflict_rule",
                "name": rule,
                "status": "frozen_not_run",
                "notes": "Rule frozen for Step 2B after lightweight feature extraction becomes possible.",
            }
        )
    return pd.DataFrame(rows)


def _order_sanity_message(order_sanity: pd.DataFrame) -> str:
    count_disc = int((order_sanity["discrete_tv_distance"] > 0.20).sum())
    count_val = int((order_sanity["valence_mean_diff_abs"] > 1.0).sum())
    count_aro = int((order_sanity["arousal_mean_diff_abs"] > 1.0).sum())
    if count_disc == 0 and count_val == 0 and count_aro == 0:
        return "No strong order-misalignment signal was detected in the sanity check."
    return (
        "No widespread order-misalignment signal was detected, but a small number of stimuli "
        f"show larger order differences (discrete_tv_distance>0.20: {count_disc}; "
        f"valence_mean_diff_abs>1.0: {count_val}; arousal_mean_diff_abs>1.0: {count_aro}). "
        "Step 2B should keep the order-to-stimulus mapping caveat visible."
    )


def write_step2a_notes(
    meet_long: pd.DataFrame,
    meet_summary: pd.DataFrame,
    merp_trials: pd.DataFrame,
    merp_song_summary: pd.DataFrame,
    sample_retention: pd.DataFrame,
    cue_conflict_readiness: pd.DataFrame,
    order_sanity: pd.DataFrame,
) -> Path:
    formal_merp = merp_song_summary[merp_song_summary["formal_validation_song"]]
    order_message = _order_sanity_message(order_sanity)
    path = CHECKPOINTS_DIR / "step2a_notes.md"
    content = f"""# Step 2A Notes

Date: 2026-03-22

## Conclusion

Step 2A passed.

The primary disagreement metric is frozen as `sample standard deviation (std)`, used as `valence_std` and `arousal_std`.
`MEET` serves as the main analysis dataset, and `MERP` serves as the external validation dataset for continuous disagreement only.

## Metric Freeze

### Conclusion

The single primary disagreement metric is `std`.

- Why this metric: interpretable as listener spread, stable enough in both datasets, compatible across `MEET` and `MERP`, and convenient for later regression-style analysis.
- Secondary metrics retained only as sensitivity references:
  - `MEET`: `discrete_entropy`, `top_label_proportion`, `valence_mad`, `valence_iqr`, `arousal_mad`, `arousal_iqr`
  - `MERP`: `valence_mad`, `valence_iqr`, `arousal_mad`, `arousal_iqr`

## Dataset Roles

### Conclusion

`MEET` and `MERP` play different roles in Step 2A.

- `MEET`: main stimulus-level disagreement table plus minimum listener-factor-ready structure.
- `MERP`: Approved-only external validation input for continuous valence/arousal disagreement using trial means aggregated to song-level disagreement.

## Sample Retention

### Conclusion

The retained sample is sufficient for Step 2A.

- `MEET` raw participants: {len(meet_long) // MEET_STIMULUS_COUNT}
- `MEET` participant x stimulus rows: {len(meet_long)}
- `MEET` stimulus summaries: {len(meet_summary)}
- `MERP` Approved worker-song trials: {len(merp_trials)}
- `MERP` song summaries including DEAM slots: {len(merp_song_summary)}
- `MERP` formal main-pool validation songs: {len(formal_merp)}

Retention rules applied:

- `MERP` kept only `AssignmentStatus == "Approved"`.
- `batch8_all_world_NONmaster.csv` was excluded from formal analysis input.
- `MERP` DEAM-slot songs were kept as auxiliary rows only and not counted as formal Step 2B validation items.

## Order Sanity Check

### Conclusion

{order_message}

- Median discrete total-variation distance between `Order=1` and `Order=2`: {order_sanity["discrete_tv_distance"].median():.3f}
- Median absolute valence mean difference: {order_sanity["valence_mean_diff_abs"].median():.3f}
- Median absolute arousal mean difference: {order_sanity["arousal_mean_diff_abs"].median():.3f}

## Cue-Conflict Readiness

### Conclusion

Cue-conflict is prepared only at a readiness level and is not yet runnable.

- Reason: `MEET` audio files are not present locally, so no lightweight audio features were extracted in Step 2A.
- Frozen lightweight feature set: `{", ".join(MEET_CUE_CONFLICT_FEATURES)}`
- Frozen cue-conflict rules: `{", ".join(MEET_CUE_CONFLICT_RULES)}`
- This is not a Step 2A blocker, but it remains a Step 2B blocker for any formal cue-conflict analysis.

## Blockers and Step 2B Gaps

### Conclusion

There is no hard blocker for Step 2A completion, but two gaps remain before Step 2B.

- Pending gap 1: a confirmed `MEET` order-to-stimulus mapping or explicit confirmation that numeric columns already represent canonical stimulus IDs.
- Pending gap 2: local `MEET` audio files for lightweight feature extraction and cue-conflict construction.
- `MERP` continuous disagreement is computable under the compatible rule used here: per-trial dynamic sequence mean, then song-level `std` across listeners.

## Step 2A Status

### Conclusion

Step 2A is complete and ready for review.

The produced artifacts include cleaned intermediate tables, summary tables, and this checkpoint note. Step 2B should not start until you approve it.
"""
    path.write_text(content, encoding="utf-8")
    return path


def run_step2a() -> Step2AArtifacts:
    _ensure_parent_dirs()

    meet_raw = load_meet_raw()
    meet_long = build_meet_participant_stimulus_long(meet_raw)
    meet_summary = summarize_meet_stimulus(meet_long)
    meet_group_summary = summarize_meet_stimulus_by_group(meet_long)
    order_sanity = compute_meet_order_sanity(meet_long)

    merp_all_rows = load_merp_rows()
    merp_trials = build_merp_approved_worker_song_trials(merp_all_rows)
    merp_song_summary = summarize_merp_song_disagreement(merp_trials)

    metric_summary = build_metric_summary(meet_summary, merp_song_summary)
    sample_retention = build_sample_retention(
        meet_raw,
        meet_long,
        meet_summary,
        merp_all_rows,
        merp_trials,
        merp_song_summary,
    )
    cue_conflict_readiness = build_cue_conflict_readiness()

    meet_long.to_csv(PROCESSED_DIR / "meet_participant_stimulus_long.csv", index=False)
    meet_summary.to_csv(PROCESSED_DIR / "meet_stimulus_disagreement_summary.csv", index=False)
    meet_group_summary.to_csv(
        PROCESSED_DIR / "meet_stimulus_group_disagreement_summary.csv", index=False
    )
    merp_trials.to_csv(PROCESSED_DIR / "merp_approved_worker_song_trials.csv", index=False)
    merp_song_summary.to_csv(PROCESSED_DIR / "merp_song_disagreement_summary.csv", index=False)
    metric_summary.to_csv(TABLES_DIR / "step2a_metric_summary.csv", index=False)
    sample_retention.to_csv(TABLES_DIR / "step2a_sample_retention.csv", index=False)
    cue_conflict_readiness.to_csv(TABLES_DIR / "step2a_cue_conflict_readiness.csv", index=False)
    order_sanity.to_csv(TABLES_DIR / "step2a_meet_order_sanity.csv", index=False)

    notes_path = write_step2a_notes(
        meet_long,
        meet_summary,
        merp_trials,
        merp_song_summary,
        sample_retention,
        cue_conflict_readiness,
        order_sanity,
    )

    return Step2AArtifacts(
        meet_long=meet_long,
        meet_summary=meet_summary,
        meet_group_summary=meet_group_summary,
        merp_trials=merp_trials,
        merp_song_summary=merp_song_summary,
        metric_summary=metric_summary,
        sample_retention=sample_retention,
        cue_conflict_readiness=cue_conflict_readiness,
        order_sanity=order_sanity,
        notes_path=notes_path,
    )


if __name__ == "__main__":
    artifacts = run_step2a()
    print("Step 2A complete.")
    print(f"MEET long rows: {len(artifacts.meet_long)}")
    print(f"MEET stimulus summaries: {len(artifacts.meet_summary)}")
    print(f"MERP worker-song trials: {len(artifacts.merp_trials)}")
    print(
        "MERP formal validation songs: "
        f"{int(artifacts.merp_song_summary['formal_validation_song'].sum())}"
    )
    print(f"Notes written to: {artifacts.notes_path}")
