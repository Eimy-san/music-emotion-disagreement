from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

BASE_FONT_SIZE = 10
PANEL_LABEL_SIZE = 11
TICK_FONT_SIZE = 9
LINE_WIDTH = 0.9
POINT_EDGE_WIDTH = 0.5
FIGURE_DPI = 600
ANNOTATION_ALPHA = 0.92

MEET_COLORS = {
    "musician": "#1f4e79",
    "non_musician": "#9a9a9a",
}

MERP_COLORS = {
    "Yes": "#167f6b",
    "No": "#9a9a9a",
}

NEUTRAL_COLORS = {
    "connector": "#d0d0d0",
    "annotation_box": "#f7f7f7",
    "text": "#202020",
}


def apply_publication_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": TICK_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "axes.linewidth": LINE_WIDTH,
            "xtick.major.width": LINE_WIDTH,
            "ytick.major.width": LINE_WIDTH,
            "xtick.minor.width": LINE_WIDTH,
            "ytick.minor.width": LINE_WIDTH,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH)
    ax.tick_params(width=LINE_WIDTH, length=4, color=NEUTRAL_COLORS["text"])


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.01,
        1.025,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=NEUTRAL_COLORS["text"],
    )


def add_annotation_box(ax: plt.Axes, text: str, *, loc: str = "upper right") -> None:
    if loc == "upper right":
        x_pos, y_pos, ha = 0.98, 0.98, "right"
    else:
        x_pos, y_pos, ha = 0.03, 0.98, "left"

    ax.text(
        x_pos,
        y_pos,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=TICK_FONT_SIZE,
        color=NEUTRAL_COLORS["text"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": (247 / 255, 247 / 255, 247 / 255, ANNOTATION_ALPHA),
            "edgecolor": "#d9d9d9",
            "linewidth": 0.6,
        },
    )


def save_figure_all_formats(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    fig.savefig(out_base.with_suffix(".png"), dpi=FIGURE_DPI)
