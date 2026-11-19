from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import DEFAULT_PLOT_STYLE


def apply_plot_style() -> None:
    sns.set_theme(
        style=DEFAULT_PLOT_STYLE["style"],
        palette=DEFAULT_PLOT_STYLE["palette"],
        context=DEFAULT_PLOT_STYLE["context"],
    )


def save_disagreement_hist(df: pd.DataFrame, value_col: str, out_path: Path) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[value_col].dropna(), bins=20, kde=True, ax=ax)
    ax.set_title("Disagreement distribution")
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_group_boxplot(df: pd.DataFrame, group_col: str, value_col: str, out_path: Path) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
    sns.stripplot(data=df, x=group_col, y=value_col, color="black", alpha=0.4, ax=ax)
    ax.set_title(f"{value_col} by {group_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_scatter_with_fit(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={"alpha": 0.7}, ax=ax)
    ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_effect_direction_plot(df: pd.DataFrame, effect_col: str, label_col: str, out_path: Path) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.45)))
    ordered = df.sort_values(effect_col)
    ax.barh(ordered[label_col], ordered[effect_col])
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title("Effect direction comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

