from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import OUTPUTS_DIR, PROJECT_ROOT
from .manuscript_plot_style import (
    BASE_FONT_SIZE,
    FIGURE_DPI,
    MEET_COLORS,
    MERP_COLORS,
    NEUTRAL_COLORS,
    POINT_EDGE_WIDTH,
    TICK_FONT_SIZE,
    add_annotation_box,
    add_panel_label,
    apply_publication_style,
    save_figure_all_formats,
    style_axis,
)

FIGURES_DIR = OUTPUTS_DIR / "figures"
MAIN_FIGURES_DIR = FIGURES_DIR / "main"
PANEL_FIGURES_DIR = FIGURES_DIR / "panels"
MANUSCRIPT_TABLES_DIR = OUTPUTS_DIR / "tables" / "manuscript"


def _format_p_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def _format_estimate(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def _markdown_table(df: pd.DataFrame) -> str:
    safe_df = df.copy()
    safe_df = safe_df.fillna("")
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in safe_df.to_numpy().tolist()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def _render_table_markdown(title: str, df: pd.DataFrame, note: str = "") -> str:
    parts = [f"# {title}", "", _markdown_table(df).rstrip()]
    if note:
        parts.extend(["", f"Note. {note}"])
    return "\n".join(parts) + "\n"


def _ensure_dirs() -> None:
    for path in (MAIN_FIGURES_DIR, PANEL_FIGURES_DIR, MANUSCRIPT_TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / rel_path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _annotation_from_result(result_df: pd.DataFrame, outcome: str, *, prefix: str, sample_note: str = "") -> str:
    row = result_df.loc[result_df["outcome"] == outcome].iloc[0]
    lines = [
        f"{prefix} = {_format_estimate(float(row['estimate']))}",
        f"p = {_format_p_value(float(row['p_value']))}",
        f"n = {int(row['n'])}",
    ]
    if sample_note:
        lines.append(sample_note)
    return "\n".join(lines)


def _plot_meet_dumbbell(ax: plt.Axes, meet_wide: pd.DataFrame, metric: str, annotation_text: str, panel_label: str) -> None:
    plot_df = meet_wide.sort_values("midpoint", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    for _, row in plot_df.iterrows():
        ax.hlines(
            y=row["rank"],
            xmin=min(row["musician"], row["non_musician"]),
            xmax=max(row["musician"], row["non_musician"]),
            color=NEUTRAL_COLORS["connector"],
            linewidth=0.65,
            alpha=0.58,
            zorder=1,
        )

    ax.scatter(
        plot_df["non_musician"],
        y_pos,
        s=14,
        color=MEET_COLORS["non_musician"],
        edgecolor="white",
        linewidth=POINT_EDGE_WIDTH,
        zorder=3,
    )
    ax.scatter(
        plot_df["musician"],
        y_pos,
        s=14,
        color=MEET_COLORS["musician"],
        edgecolor="white",
        linewidth=POINT_EDGE_WIDTH,
        zorder=4,
    )

    ax.set_ylabel("Stimuli (sorted)")
    readable_metric = "Valence disagreement (SD)" if metric == "valence_std" else "Arousal disagreement (SD)"
    ax.set_xlabel(readable_metric)
    ax.set_yticks([])
    ax.set_ylim(-3, len(plot_df) + 5.5)
    ax.margins(x=0.02)
    style_axis(ax)
    add_panel_label(ax, panel_label)
    add_annotation_box(ax, annotation_text, loc="upper left")


def _plot_merp_distribution(
    ax: plt.Axes,
    merp_df: pd.DataFrame,
    metric: str,
    annotation_text: str,
    panel_label: str,
) -> None:
    order = ["No", "Yes"]
    palette = [MERP_COLORS["No"], MERP_COLORS["Yes"]]
    label_map = {
        "No": f"No\n(n={int((merp_df['training'] == 'No').sum())})",
        "Yes": f"Yes\n(n={int((merp_df['training'] == 'Yes').sum())})",
    }

    sns.violinplot(
        data=merp_df,
        x="training",
        y=metric,
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.78,
        width=0.72,
        ax=ax,
    )
    for collection in ax.collections[: len(order)]:
        collection.set_alpha(0.88)

    sns.boxplot(
        data=merp_df,
        x="training",
        y=metric,
        order=order,
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#404040", "linewidth": 0.8, "zorder": 4},
        whiskerprops={"linewidth": 0.8},
        medianprops={"color": "#202020", "linewidth": 0.9},
        ax=ax,
    )
    sns.stripplot(
        data=merp_df,
        x="training",
        y=metric,
        order=order,
        color="#404040",
        alpha=0.22,
        size=1.8,
        jitter=0.15,
        ax=ax,
        zorder=5,
    )

    readable_metric = {
        "mean_valence_abs_dev": "Mean valence deviation from consensus",
        "mean_arousal_abs_dev": "Mean arousal deviation from consensus",
    }[metric]
    ax.set_xlabel("Training")
    ax.set_ylabel(readable_metric)
    ax.set_xticklabels([label_map[key] for key in order])
    style_axis(ax)
    add_panel_label(ax, panel_label)
    add_annotation_box(ax, annotation_text)


def _plot_effect_summary_panel(
    ax: plt.Axes,
    *,
    rows: list[dict[str, object]],
    x_label: str,
    panel_label: str,
    x_limits: tuple[float, float],
    panel_annotation: str,
) -> None:
    y_pos = np.arange(len(rows))[::-1]
    labels = [str(row["label"]) for row in rows]
    effects = [float(row["estimate"]) for row in rows]
    colors = [str(row["color"]) for row in rows]

    ax.axvline(0.0, color="#7a7a7a", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    for y in y_pos:
        ax.hlines(y, x_limits[0], x_limits[1], color="#ececec", linewidth=0.8, zorder=0)
    for y, effect, color in zip(y_pos, effects, colors):
        ax.scatter(effect, y, s=54, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    style_axis(ax)
    add_panel_label(ax, panel_label)
    add_annotation_box(ax, panel_annotation, loc="upper left")


def _build_meet_wide(meet_group: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivoted = (
        meet_group.pivot(index="stimulus_id", columns="group_label", values=metric)
        .rename_axis(columns=None)
        .reset_index()
    )
    pivoted["midpoint"] = (pivoted["musician"] + pivoted["non_musician"]) / 2.0
    pivoted = pivoted.sort_values("midpoint", ascending=True).reset_index(drop=True)
    pivoted["rank"] = np.arange(len(pivoted))
    return pivoted


def generate_figures() -> None:
    apply_publication_style()
    meet_group = _load_csv("data/processed/meet_stimulus_group_disagreement_summary.csv")
    meet_results = _load_csv("outputs/tables/step2b1_meet_core_results.csv")
    merp_listener = _load_csv("data/processed/merp_listener_validation_table.csv")
    merp_results = _load_csv("outputs/tables/step2b1_merp_validation_results.csv")

    meet_metrics = ["valence_std", "arousal_std"]
    meet_wide_map = {metric: _build_meet_wide(meet_group, metric) for metric in meet_metrics}
    meet_min = min(df[["musician", "non_musician"]].min().min() for df in meet_wide_map.values())
    meet_max = max(df[["musician", "non_musician"]].max().max() for df in meet_wide_map.values())
    meet_pad = (meet_max - meet_min) * 0.06

    fig1, axes1 = plt.subplots(1, 2, figsize=(13.2, 11.7), sharey=True)
    for ax, metric, panel in zip(axes1, meet_metrics, ["A", "B"]):
        annotation = _annotation_from_result(
            meet_results,
            metric,
            prefix="Paired mean diff",
        )
        _plot_meet_dumbbell(ax, meet_wide_map[metric], metric, annotation, panel)
        ax.set_xlim(meet_min - meet_pad, meet_max + meet_pad)
    fig1.subplots_adjust(left=0.08, right=0.985, top=0.985, bottom=0.07, wspace=0.18)
    save_figure_all_formats(fig1, MAIN_FIGURES_DIR / "figure1_meet_main_results")

    for metric, panel in zip(meet_metrics, ["A", "B"]):
        fig, ax = plt.subplots(figsize=(6.2, 11.0))
        annotation = _annotation_from_result(meet_results, metric, prefix="Paired mean diff")
        _plot_meet_dumbbell(ax, meet_wide_map[metric], metric, annotation, panel)
        ax.set_xlim(meet_min - meet_pad, meet_max + meet_pad)
        save_figure_all_formats(fig, PANEL_FIGURES_DIR / f"figure1{panel}_meet_{metric}")
        plt.close(fig)

    merp_metrics = ["mean_valence_abs_dev", "mean_arousal_abs_dev"]
    merp_min = min(merp_listener[metric].min() for metric in merp_metrics)
    merp_max = max(merp_listener[metric].max() for metric in merp_metrics)
    merp_pad = (merp_max - merp_min) * 0.08

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, metric, panel in zip(axes2, merp_metrics, ["A", "B"]):
        annotation = _annotation_from_result(
            merp_results,
            metric,
            prefix="Mean diff (Yes - No)",
        )
        _plot_merp_distribution(ax, merp_listener, metric, annotation, panel)
        ax.set_ylim(merp_min - merp_pad, merp_max + merp_pad)
    fig2.subplots_adjust(left=0.065, right=0.985, top=0.96, bottom=0.14, wspace=0.22)
    save_figure_all_formats(fig2, MAIN_FIGURES_DIR / "figure2_merp_external_validation")

    merp_panel_names = {
        "mean_valence_abs_dev": "figure2A_merp_valence_dev",
        "mean_arousal_abs_dev": "figure2B_merp_arousal_dev",
    }
    for metric, panel in zip(merp_metrics, ["A", "B"]):
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        annotation = _annotation_from_result(
            merp_results,
            metric,
            prefix="Mean diff (Yes - No)",
        )
        _plot_merp_distribution(ax, merp_listener, metric, annotation, panel)
        ax.set_ylim(merp_min - merp_pad, merp_max + merp_pad)
        save_figure_all_formats(fig, PANEL_FIGURES_DIR / merp_panel_names[metric])
        plt.close(fig)

    meet_summary_rows = []
    for outcome, label in [("valence_std", "Valence SD"), ("arousal_std", "Arousal SD")]:
        row = meet_results.loc[
            (meet_results["section"] == "stimulus_level_group_comparison") & (meet_results["outcome"] == outcome)
        ].iloc[0]
        meet_summary_rows.append(
            {
                "label": label,
                "estimate": float(row["estimate"]),
                "color": MEET_COLORS["musician"],
            }
        )

    merp_summary_rows = []
    for outcome, label in [
        ("mean_valence_abs_dev", "Valence deviation"),
        ("mean_arousal_abs_dev", "Arousal deviation"),
    ]:
        row = merp_results.loc[
            (merp_results["section"] == "worker_level_group_comparison") & (merp_results["outcome"] == outcome)
        ].iloc[0]
        merp_summary_rows.append(
            {
                "label": label,
                "estimate": float(row["estimate"]),
                "color": MERP_COLORS["Yes"],
            }
        )

    meet_summary_note = "Saved point estimates only\nn = 116 stimuli"
    merp_summary_note = "Saved point estimates only\nn = 259 workers"

    fig3, axes3 = plt.subplots(1, 2, figsize=(11.8, 3.8))
    _plot_effect_summary_panel(
        axes3[0],
        rows=meet_summary_rows,
        x_label="Paired mean difference (musician - non-musician)",
        panel_label="A",
        x_limits=(-0.01, 0.115),
        panel_annotation=meet_summary_note,
    )
    _plot_effect_summary_panel(
        axes3[1],
        rows=merp_summary_rows,
        x_label="Mean difference (Yes - No)",
        panel_label="B",
        x_limits=(-0.085, 0.015),
        panel_annotation=merp_summary_note,
    )
    fig3.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.19, wspace=0.34)
    save_figure_all_formats(fig3, MAIN_FIGURES_DIR / "figure3_cross_dataset_effect_summary")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_effect_summary_panel(
        ax,
        rows=meet_summary_rows,
        x_label="Paired mean difference (musician - non-musician)",
        panel_label="A",
        x_limits=(-0.01, 0.115),
        panel_annotation=meet_summary_note,
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.94, bottom=0.19)
    save_figure_all_formats(fig, PANEL_FIGURES_DIR / "figure3A_meet_effect_summary")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _plot_effect_summary_panel(
        ax,
        rows=merp_summary_rows,
        x_label="Mean difference (Yes - No)",
        panel_label="B",
        x_limits=(-0.085, 0.015),
        panel_annotation=merp_summary_note,
    )
    fig.subplots_adjust(left=0.23, right=0.985, top=0.94, bottom=0.19)
    save_figure_all_formats(fig, PANEL_FIGURES_DIR / "figure3B_merp_effect_summary")
    plt.close(fig)

    plt.close(fig3)

    plt.close(fig1)
    plt.close(fig2)


def _extract_primary_variables(variable_freeze: pd.DataFrame, dataset: str, item_type: str) -> Iterable[str]:
    subset = variable_freeze[
        (variable_freeze["dataset"] == dataset)
        & (variable_freeze["item_type"] == item_type)
        & (variable_freeze["freeze_status"] == "primary")
    ]
    return subset["name"].tolist()


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_tier = _load_csv("outputs/tables/result_tier_table.csv")
    variable_freeze = _load_csv("outputs/tables/variable_freeze_table.csv")
    sample_retention = _load_csv("outputs/tables/step2a_sample_retention.csv")
    metric_summary = _load_csv("outputs/tables/step2a_metric_summary.csv")
    meet_results = _load_csv("outputs/tables/step2b1_meet_core_results.csv")
    merp_results = _load_csv("outputs/tables/step2b1_merp_validation_results.csv")
    merp_listener = _load_csv("data/processed/merp_listener_validation_table.csv")

    meet_participants = int(
        sample_retention.loc[
            (sample_retention["dataset"] == "MEET") & (sample_retention["stage"] == "raw_participants"),
            "n_rows",
        ].iloc[0]
    )
    meet_items = int(
        metric_summary.loc[metric_summary["dataset"] == "MEET", "n_formal_items"].iloc[0]
    )
    merp_workers = int(merp_listener["worker_id"].nunique())
    merp_items = int(
        sample_retention.loc[
            (sample_retention["dataset"] == "MERP") & (sample_retention["stage"] == "song_summary_formal_main_pool"),
            "n_unique_items",
        ].iloc[0]
    )

    table1 = pd.DataFrame(
        [
            {
                "Dataset": "MEET",
                "Role": "Main analysis",
                "Analysis unit": "Stimulus-level main; participant-level supporting",
                "Retained sample": f"{meet_participants} participants",
                "Formal items": f"{meet_items} stimuli",
                "Primary outcomes": "valence_std; arousal_std",
                "Primary listener variable": "group",
                "Key rule": "Discrete outcomes reported only as supporting evidence",
            },
            {
                "Dataset": "MERP",
                "Role": "Cautious external support",
                "Analysis unit": "Worker-level deviation outcomes",
                "Retained sample": f"{merp_workers} workers",
                "Formal items": f"{merp_items} songs",
                "Primary outcomes": "mean_valence_abs_dev; mean_arousal_abs_dev",
                "Primary listener variable": "training",
                "Key rule": "Approved-only worker pool; no discrete-outcome replication",
            },
        ]
    )

    meet_main = meet_results[
        (meet_results["section"] == "stimulus_level_group_comparison")
        & (meet_results["outcome"].isin(["valence_std", "arousal_std"]))
    ].copy()
    meet_main["Dataset"] = "MEET"
    meet_main["Analysis level"] = "Stimulus-level"

    merp_main = merp_results[
        (merp_results["section"] == "worker_level_group_comparison")
        & (merp_results["outcome"].isin(["mean_valence_abs_dev", "mean_arousal_abs_dev"]))
    ].copy()
    merp_main["Dataset"] = "MERP"
    merp_main["Analysis level"] = "Worker-level"

    meet_main["Contrast"] = "Musician - non-musician"
    meet_main["Group A"] = "Musician"
    meet_main["Group B"] = "Non-musician"
    merp_main["Contrast"] = "Yes - No"
    merp_main["Group A"] = "Yes"
    merp_main["Group B"] = "No"

    table2 = pd.concat(
        [
            meet_main[
                [
                    "Dataset",
                    "Analysis level",
                    "outcome",
                    "Contrast",
                    "Group A",
                    "group_a_mean",
                    "Group B",
                    "group_b_mean",
                    "estimate",
                    "p_value",
                    "n",
                ]
            ],
            merp_main[
                [
                    "Dataset",
                    "Analysis level",
                    "outcome",
                    "Contrast",
                    "Group A",
                    "group_a_mean",
                    "Group B",
                    "group_b_mean",
                    "estimate",
                    "p_value",
                    "n",
                ]
            ],
        ],
        ignore_index=True,
    )
    table2["group_a_mean"] = table2["group_a_mean"].map(_format_estimate)
    table2["group_b_mean"] = table2["group_b_mean"].map(_format_estimate)
    table2["estimate"] = table2["estimate"].map(_format_estimate)
    table2["p_value"] = table2["p_value"].map(_format_p_value)
    table2["n"] = table2["n"].astype(int)
    table2 = table2.rename(
        columns={
            "outcome": "Outcome",
            "group_a_mean": "Mean A",
            "group_b_mean": "Mean B",
            "estimate": "Estimate",
            "p_value": "p value",
            "n": "n",
        }
    )

    supplementary = result_tier[
        result_tier["status_for_writing"].isin(["supporting_text", "appendix_or_note", "not_for_main_text"])
    ].rename(
        columns={
            "dataset": "Dataset",
            "outcome": "Finding",
            "predictor": "Predictor",
            "tier": "Status",
            "status_for_writing": "Reporting location",
            "reason": "Rationale",
        }
    )[["Dataset", "Finding", "Predictor", "Status", "Reporting location", "Rationale"]].copy()
    supplementary["Reporting location"] = supplementary["Reporting location"].replace(
        {
            "supporting_text": "Brief supporting text",
            "appendix_or_note": "Supplementary reporting only",
            "not_for_main_text": "Not emphasized in main text",
        }
    )
    supplementary["Status"] = supplementary["Status"].replace(
        {
            "robust": "Secondary but robust",
            "directional_but_weak": "Directional but weak",
            "exploratory_not_for_main_text": "Exploratory",
            "exploratory": "Exploratory",
            "excluded": "Excluded from interpretation",
        }
    )

    additional_rows = []
    variable_subset = variable_freeze[
        (variable_freeze["item_type"] == "listener_factor")
        & (variable_freeze["writing_role"].isin(["supporting_text", "not_for_main_text"]))
    ]
    existing_names = set(supplementary["Predictor"].astype(str))
    for _, row in variable_subset.iterrows():
        if str(row["name"]) in existing_names:
            continue
        additional_rows.append(
            {
                "Dataset": row["dataset"],
                "Finding": "Secondary listener-related variable",
                "Predictor": row["name"],
                "Status": row["freeze_status"],
                "Reporting location": row["writing_role"],
                "Rationale": row["reason"],
            }
        )
    supplementary = pd.concat([supplementary, pd.DataFrame(additional_rows)], ignore_index=True)
    supplementary["Reporting location"] = supplementary["Reporting location"].replace(
        {
            "supporting_text": "Brief supporting text",
            "not_for_main_text": "Not emphasized in main text",
        }
    )
    supplementary["Status"] = supplementary["Status"].replace(
        {
            "exploratory": "Exploratory",
            "excluded": "Excluded from interpretation",
        }
    )
    supplementary["Rationale"] = supplementary["Rationale"].replace(
        {
            "Useful cross-check, but secondary because the project's cross-dataset bridge is continuous disagreement.": "Useful cross-check, but secondary because the main cross-dataset comparison relies on continuous disagreement outcomes.",
            "Conceptually relevant but weak in Step 2B1 and partly overlapping with group/musical_experience.": "Conceptually relevant but weak in the saved analyses and partly overlapping with group or musical_experience.",
            "Conceptually relevant but weak in Step 2B1 and partly overlapping with group/years_musical_education.": "Conceptually relevant but weak in the saved analyses and partly overlapping with group or years_musical_education.",
            "Retained as a light control, but no clear Step 2B1 signal.": "Retained as a light control, but no clear signal in the saved analyses.",
            "Weak and unstable in Step 2B1; keep out of main written claims.": "Weak and unstable in the saved analyses; keep out of main written claims.",
        }
    )
    return table1, table2, supplementary


def write_tables(table1: pd.DataFrame, table2: pd.DataFrame, supplementary: pd.DataFrame) -> None:
    outputs = [
        (
            "table1_study_design",
            table1,
            "Table 1. Study design, retained samples, analysis units, and primary variables.",
            "MEET provides the main stimulus-level analysis. MERP is retained only as cautious external support and is not treated as a clean replication.",
        ),
        (
            "table2_main_results",
            table2,
            "Table 2. Numerical summary of the primary results retained for the main text.",
            "Point estimates and p values are reproduced from the saved comparison-result tables. Confidence intervals are not shown because they were not included in the frozen result exports.",
        ),
        (
            "supplementary_table_s1_result_placement",
            supplementary,
            "Supplementary Table S1. Secondary and exploratory findings.",
            "This table lists secondary, exploratory, or de-emphasized findings that are retained for transparency but are not foregrounded in the main text.",
        ),
    ]
    for name, df, title, note in outputs:
        df.to_csv(MANUSCRIPT_TABLES_DIR / f"{name}.csv", index=False)
        _write_text(MANUSCRIPT_TABLES_DIR / f"{name}.md", _render_table_markdown(title, df, note))


def write_notes() -> None:
    figure_generation_note = """# Figure Generation Note

## Conclusion

本次主文图表包已精修为 3 张主图、2 张主表和 1 张补充表。按新的主文排位，当前结果图将作为 Figure 2、Figure 3 和 Figure 4 使用，其中 Figure 4 已进一步收紧为更接近期刊常见结果摘要图的 zero-line point-estimate summary figure，用于总结而不是重复原始分布。

## Figure 2: MEET main results

- Source tables:
  - `outputs/tables/step2b1_meet_core_results.csv`
  - `data/processed/meet_stimulus_group_disagreement_summary.csv`
- Why retained:
  - This is the clearest and strongest main result in the current project state.
  - It directly visualizes the two frozen stimulus-level primary outcomes.
- Simplification choices:
  - Only `valence_std` and `arousal_std` were retained.
  - `discrete_entropy` was intentionally omitted from the main figure and left to supporting material.
  - A paired dumbbell design was used because the comparison is stimulus-level and paired by item.
  - Statistical annotations were moved to the upper-left corner to avoid covering the densest data region.
  - Point size, connector weight, annotation opacity, and top margin were refined for publication use.

## Figure 3: MERP external validation

- Source tables:
  - `outputs/tables/step2b1_merp_validation_results.csv`
  - `data/processed/merp_listener_validation_table.csv`
- Why retained:
  - This is the most stable and interpretable MERP result.
  - It supports the cautious cross-dataset claim without overstating replication.
- Simplification choices:
  - Only the training contrast on the two frozen participant-level primary outcomes was retained.
  - Age and OLS details were omitted from the main figure and left to supporting material.
  - A violin + box + jitter design was used because the MERP comparison is not paired.
  - The violin / box / jitter overlay was visually thinned to reduce crowding while preserving distribution information.

## Figure 4: Cross-dataset effect summary

- Source tables:
  - `outputs/tables/step2b1_meet_core_results.csv`
  - `outputs/tables/step2b1_merp_validation_results.csv`
- Why retained:
  - It gives a mature result-synthesis view of the main text findings without repeating the raw stimulus-level or worker-level distributions.
  - It is more valuable than a study overview figure because it directly supports the manuscript's central claim.
- Simplification choices:
  - Only the four main-text continuous results were retained.
  - No weak or exploratory result was promoted into this figure.
  - The figure uses separate dataset-specific panels because the contrasts are not on the same analysis level.
  - The figure was tightened into a forest-style point-estimate summary with a zero reference line.
  - Confidence intervals were not added because they were not available in the frozen saved result tables.
  - Exact p-values were moved to the legend draft instead of being repeated inside the figure.

## Table 1

- Source tables:
  - `outputs/tables/step2a_sample_retention.csv`
  - `outputs/tables/step2a_metric_summary.csv`
  - `outputs/tables/variable_freeze_table.csv`
- Why retained:
  - It gives the minimum design, sample, outcome, and retention context needed for Methods and Results.
- Reconstruction choices:
  - The table was shortened into compact data-first columns.
  - Long descriptive sentences were removed from the body of the table.
  - Reader-facing titles and brief notes were added so the markdown export reads like a paper table rather than an internal project dump.

## Table 2

- Source tables:
  - `outputs/tables/step2b1_meet_core_results.csv`
  - `outputs/tables/step2b1_merp_validation_results.csv`
  - `outputs/tables/result_tier_table.csv`
- Why retained:
  - It concentrates the main text results into a single summary table.
- Simplification choices:
  - Only the four main-text numerical results were retained.
  - The cross-dataset synthesis statement was removed from the table and moved to Figure 3 / manuscript narrative, so the table remains numeric and publication-friendly.
  - Confidence intervals were not added because they were not included in the saved result exports.

## Supplementary Table S1

- Source tables:
  - `outputs/tables/result_tier_table.csv`
  - `outputs/tables/variable_freeze_table.csv`
- Why retained:
  - It preserves transparency about secondary and exploratory findings without exposing internal workflow labels in the manuscript package.
  - Internal project terms were removed from the title, column names, and explanatory text.
"""

    figure_legend_draft = """# Figure Legend Draft

## Conclusion

以下图注草稿已按主文保守写法准备，可直接作为后续写稿基础。

## Figure 2

`Figure 2. Stimulus-level disagreement in MEET by listener group.`

`(A)` shows stimulus-level valence disagreement (`valence_std`) for the musician and non-musician groups across the same 116 stimuli. Each horizontal connector links the two group-specific values for one stimulus. `MEET` shows higher stimulus-level valence disagreement in the musician group than in the non-musician group.

`(B)` shows the same paired comparison for stimulus-level arousal disagreement (`arousal_std`). Across both continuous stimulus-level outcomes, the musician group exhibits higher disagreement than the non-musician group. This figure visualizes the main result of the study and should be interpreted at the stimulus level rather than the participant level.

## Figure 3

`Figure 3. Worker-level deviation from consensus in MERP by training status.`

`(A)` shows worker-level mean valence deviation from song-level consensus (`mean_valence_abs_dev`) for listeners with and without training. `MERP` indicates lower valence deviation among listeners who reported training.

`(B)` shows the same contrast for worker-level mean arousal deviation from consensus (`mean_arousal_abs_dev`). Across both continuous participant-level outcomes, training is associated with lower deviation from consensus. This figure is best interpreted as cautious external support for the claim that listener-related variables matter, rather than as a clean replication of the MEET stimulus-level group effect.

## Figure 4

`Figure 4. Summary of the main continuous results across datasets.`

`(A)` summarizes the two primary `MEET` stimulus-level effects as paired mean differences between the musician and non-musician groups. Positive values indicate higher disagreement in the musician group. Both primary `MEET` stimulus-level outcomes show effects in the same direction.

`(B)` summarizes the two primary `MERP` worker-level effects as mean differences between listeners with and without training (`Yes - No`). Negative values indicate lower deviation from consensus among trained listeners. The figure is presented as a point-estimate summary with a zero reference line; confidence intervals are not shown because they were not preserved in the frozen result exports. Together, the two panels provide a compact synthesis of the main results and support the interpretation that listener-related variables matter, but only partially and in a dataset-dependent way.
"""

    plotting_style_note = f"""# Plotting Style Note

## Conclusion

本次主文图表统一采用白底、无装饰、期刊友好的 publication style，目标是保证学术清晰度、复现性和后续补图一致性。

## Global style

- Font family: `Times New Roman` with `DejaVu Serif` fallback
- Base font size: `{BASE_FONT_SIZE}`
- Tick label size: `{TICK_FONT_SIZE}`
- Output formats: `PDF`, `SVG`, `PNG`
- PNG resolution: `{FIGURE_DPI} dpi`
- Background: white
- Grid lines: off by default
- Top and right spines: removed
- Figure titles: not used inside figures
- Panel labels: `(A)`, `(B)` in bold, upper-left
- Annotation boxes: compact rounded boxes with light opacity and no heavy borders

## Color system

- `MEET` musician: `{MEET_COLORS['musician']}`
- `MEET` non-musician: `{MEET_COLORS['non_musician']}`
- `MERP` training = Yes: `{MERP_COLORS['Yes']}`
- `MERP` training = No: `{MERP_COLORS['No']}`
- Connector / neutral elements: `{NEUTRAL_COLORS['connector']}`

## Visual rules

- Main figures use multi-panel layouts with shared visual language.
- Statistical information is shown only as compact annotations, not as decorative significance stars.
- Raw observations are shown only where they improve interpretability without crowding the figure.
- Exploratory and weak results are excluded from main figures and kept for supplementary placement only.

## Layout rules

- `Figure 2` uses a horizontal 1x2 panel layout with paired stimulus-level dumbbell plots.
- `Figure 3` uses a horizontal 1x2 panel layout with independent-group distribution plots.
- `Figure 4` uses a horizontal 1x2 panel layout with zero-line point-estimate summary panels.
- Each panel is also saved separately to support later manual composition if needed.

## Naming rules

- Main figures: `outputs/figures/main/figure*.pdf|svg|png`
- Panel figures: `outputs/figures/panels/figure*A|B*.pdf|svg|png`
- Manuscript tables: `outputs/tables/manuscript/*.csv` and `*.md`
"""

    _write_text(FIGURES_DIR / "figure_generation_note.md", figure_generation_note)
    _write_text(FIGURES_DIR / "figure_legend_draft.md", figure_legend_draft)
    _write_text(FIGURES_DIR / "plotting_style_note.md", plotting_style_note)


def main() -> None:
    _ensure_dirs()
    generate_figures()
    table1, table2, supplementary = build_tables()
    write_tables(table1, table2, supplementary)
    write_notes()
    print("Manuscript figure package generated.")
    print(f"Main figures: {MAIN_FIGURES_DIR}")
    print(f"Panel figures: {PANEL_FIGURES_DIR}")
    print(f"Tables: {MANUSCRIPT_TABLES_DIR}")


if __name__ == "__main__":
    main()
