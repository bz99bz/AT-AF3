"""Reproduce a clean, three-panel AT-AF3 chirality figure.

Run: python AT_AF3_chirality_redesign.py --output-dir figures
Dependencies: numpy and matplotlib.

Data are the target-level means supplied in the manuscript tables, not
individual inference seeds. Panel a uses an unweighted mean of target means.
Panels b/c retain all seven All-D and nine Local-D targets in the supplied order.
No seed distributions, confidence intervals, or error bars are reconstructed.
Small vertical marker offsets distinguish methods with nearly identical values;
the horizontal coordinates always equal the supplied accuracy values.

Names: AF3 is the baseline. AT-AF3 w/o correction omits geometric correction.
AT-AF3 is the complete method, including deterministic geometric correction.
SVG text stays editable; PDF embeds fonts. All data and styling are in this file.
"""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# (target, AF3 mean, AT-AF3 w/o correction mean, complete AT-AF3 mean)
DATA = {
    "All-D": [
        ("7P5R_2", 0.0000, 0.5528, 0.6125),
        ("7QDI_1", 0.0000, 0.5345, 0.7583),
        ("8AVR_2", 0.0000, 0.4461, 0.5135),
        ("8BFD_2", 0.0000, 0.3414, 0.4115),
        ("8GQP_1", 0.2500, 0.5925, 0.6112),
        ("8JH6_1", 0.0000, 0.9972, 0.9983),
        ("8P6Q_2", 0.0000, 0.8442, 0.8705),
    ],
    "Local-D": [
        ("7OOJ_1", 0.0000, 0.0000, 1.0000),
        ("7P0A_3", 0.0000, 0.0000, 1.0000),
        ("7P0T_3", 0.0000, 0.0000, 1.0000),
        ("7W40_6", 0.0000, 0.0000, 1.0000),
        ("8HR3_1", 0.0000, 0.8167, 1.0000),
        ("8HR4_1", 0.0000, 0.6667, 1.0000),
        ("8U78_1", 0.0000, 0.0885, 1.0000),
        ("8UM9_1", 0.0000, 0.0000, 1.0000),
        ("9BIC_1", 0.0000, 0.2625, 1.0000),
    ],
}
METHODS = ("AF3", "AT-AF3 w/o correction", "AT-AF3")
COLORS = ("#A6ADB6", "#5262A8", "#D6A34A")
MARKERS = ("o", "D", "o")
INK = "#303640"
GRID = "#E7EAF0"


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#7F8792",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def accuracy_axis(ax):
    ax.set_xlim(-0.035, 1.20)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_xlabel("Target D-chirality accuracy", labelpad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1)


def panel_title(ax, letter, title):
    ax.text(-0.025, 1.13, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=INK, va="bottom")
    ax.text(0.095, 1.135, title, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom")


def draw_summary(ax):
    means = {task: np.array([r[1:] for r in rows]).mean(axis=0)
             for task, rows in DATA.items()}
    locations = {"All-D": [6.2, 5.45, 4.7], "Local-D": [2.8, 2.05, 1.3]}
    headings = {"All-D": 7.05, "Local-D": 3.65}
    for task, values in means.items():
        ax.text(0, headings[task], f"{task} ({len(DATA[task])} targets)",
                fontsize=7.6, fontweight="bold", va="center")
        for idx, value in enumerate(values):
            y = locations[task][idx]
            ax.barh(y, value, height=0.52, color=COLORS[idx], linewidth=0)
            if value == 0:
                ax.plot(0, y, marker="|", markersize=5.5,
                        color=COLORS[idx], markeredgewidth=1.0)
            ax.text(value + 0.035, y, f"{value:.3f}", va="center",
                    fontsize=7.2, color=INK)
    ax.set_ylim(0.25, 7.7)
    ax.set_yticks([])
    accuracy_axis(ax)
    panel_title(ax, "a", "Mean accuracy")
    return means


def draw_targets(ax, task, letter):
    rows = DATA[task]
    y = np.arange(len(rows), dtype=float)
    offsets = np.array([-0.13, 0, 0.13])
    for idx, row in enumerate(rows):
        values = np.array(row[1:])
        # Identical target row, three conditions, with small categorical offsets.
        ax.plot(values, y[idx] + offsets, color="#CDD2DA", linewidth=0.8,
                zorder=2, solid_capstyle="round")
        for method in range(3):
            ax.scatter(values[method], y[idx] + offsets[method],
                       s=20 if method == 1 else 24,
                       marker=MARKERS[method], color=COLORS[method],
                       edgecolors="white", linewidths=0.35, zorder=3)
        # Preserve the four-decimal precision supplied for each target.
        ax.text(1.085, y[idx] + offsets[2], f"{values[2]:.4f}",
                fontsize=7.0, va="center", color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", pad=5)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    accuracy_axis(ax)
    panel_title(ax, letter, task)
    ax.text(1.055, -1.05, "AT-AF3", fontsize=6.7,
            ha="left", color="#676F7C", clip_on=False)


def main(output_dir):
    configure_style()
    for rows in DATA.values():
        assert all(0 <= value <= 1 for row in rows for value in row[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Approximate full-width paper canvas, with one common legend.
    fig = plt.figure(figsize=(7.45, 3.35), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.15, 1.18],
                          left=0.06, right=0.985, bottom=0.18,
                          top=0.77, wspace=0.70)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    means = draw_summary(axes[0])
    draw_targets(axes[1], "All-D", "b")
    draw_targets(axes[2], "Local-D", "c")
    legend_handles = [Line2D([], [], linestyle="none", marker=marker,
                             markersize=4.5, color=color, label=method)
                      for method, color, marker in zip(METHODS, COLORS, MARKERS)]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.51, 1.005), ncol=3, frameon=False,
               handletextpad=0.5, columnspacing=2.0, fontsize=8)
    # fig.text(0.06, 0.025,
    #          "Target-level means; 30 matched seeds per target. "
    #          "Panel a: unweighted mean across targets.",
    #          fontsize=6.6, color="#69717E")
    stem = output_dir / "AT_AF3_chirality_redesign"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=600,
                    bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(json.dumps({"macro_means": {k: v.tolist() for k, v in means.items()},
                      "target_counts": {k: len(v) for k, v in DATA.items()},
                      "outputs": [str(stem.with_suffix("." + s))
                                  for s in ("png", "svg", "pdf")]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    main(parser.parse_args().output_dir)
