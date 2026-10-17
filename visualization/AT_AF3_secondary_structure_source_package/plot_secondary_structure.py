#!/usr/bin/env python3
"""Plot the supplied target-level CHORD secondary-structure compositions.

Usage: python plot_secondary_structure.py --data secondary_structure_source.csv

Outputs: vector PDF, editable-text SVG, 600-dpi PNG, a screen preview,
and signed AT-AF3-minus-GT differences. No distributions or uncertainty are
inferred from the reported target-level percentages.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

CLASSES = ("H", "E", "C")
MODELS = ("GT", "WO", "AT")
MODEL_LABELS = {"GT": "GT", "WO": "AT-AF3\nw/o correction", "AT": "AT-AF3"}
COLORS = {"H": "#536994", "E": "#B49DC5", "C": "#DDE4EF",
          "ink": "#303746", "muted": "#6E7889", "grid": "#E4E8EF"}
TASKS = ("All-D", "Local-D")


def read_data(path: Path) -> list[dict]:
    required = {"task", "target"} | {f"{model}_{c}" for model in MODELS for c in CLASSES}
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if required - set(reader.fieldnames or []):
            raise ValueError("The source CSV is missing required columns")
        for line, row in enumerate(reader, 2):
            if row["target"] in seen:
                raise ValueError(f"Duplicate target at line {line}")
            if row["task"] not in TASKS:
                raise ValueError(f"Unknown task at line {line}")
            record = {"task": row["task"], "target": row["target"]}
            for model in MODELS:
                try:
                    values = [Decimal(row[f"{model}_{c}"]) for c in CLASSES]
                except InvalidOperation as exc:
                    raise ValueError(f"Missing or invalid percentage at line {line}") from exc
                if any(not v.is_finite() or v < 0 or v > 100 for v in values):
                    raise ValueError(f"Percentage outside 0-100 at line {line}")
                if abs(sum(values) - Decimal("100")) > Decimal("0.15"):
                    raise ValueError(f"Composition total inconsistent with rounding at line {line}")
                record[model] = values
            records.append(record)
            seen.add(row["target"])
    if len(records) != 16 or any(sum(r["task"] == task for r in records) != expected
                                 for task, expected in zip(TASKS, (7, 9))):
        raise ValueError("This CHORD figure expects the supplied seven All-D and nine Local-D targets")
    # Keep the supplied within-task order. All-D precedes Local-D.
    return [r for task in TASKS for r in records if r["task"] == task]


def differences(records: list[dict]) -> np.ndarray:
    # Decimal subtraction preserves the precision of the supplied table.
    return np.array([[float(r["AT"][j] - r["GT"][j]) for j in range(3)] for r in records])


def configure_fonts() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"], "axes.labelsize": 6.6,
        "axes.edgecolor": COLORS["muted"], "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["ink"],
        "xtick.labelsize": 6.4, "ytick.labelsize": 7,
        "xtick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 0, "axes.unicode_minus": False,
        "legend.frameon": False, "legend.fontsize": 6.8,
        "svg.fonttype": "none", "svg.hashsalt": "CHORD-secondary-structure",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def label_color(background) -> str:
    """Choose the higher-contrast foreground for an annotated heatmap cell."""
    def luminance(rgb):
        channels = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                    for v in rgb[:3]]
        return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), channels))
    bg = luminance(background)
    ink = luminance(to_rgb(COLORS["ink"]))
    white_contrast = 1.05 / (bg + 0.05)
    ink_contrast = (bg + 0.05) / (ink + 0.05)
    return "white" if white_contrast > ink_contrast else COLORS["ink"]


def figure(records: list[dict], *, scale: float = 30, label_min: float = 18.5):
    configure_fonts()
    ypos = np.r_[np.arange(7), np.arange(9) + 9]
    ylim = (17.75, -1.2)
    fig = plt.figure(figsize=(180 / 25.4, 142 / 25.4), facecolor="white")
    fig.text(0.025, 0.961, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.060, 0.960, "Secondary-structure composition", fontsize=7,
             fontweight="bold", va="top")
    fig.text(0.739, 0.961, "b", fontsize=8, fontweight="bold", va="top")
    fig.text(0.774, 0.960, "Difference from GT", fontsize=7,
             fontweight="bold", va="top")
    legend = [Patch(facecolor=COLORS[c], edgecolor="none", label=label)
              for c, label in zip(CLASSES, ("Helix (H)", "β-strand/sheet (E)", "Coil/other (C)"))]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.060, 0.917),
               ncol=3, columnspacing=1.0, handlelength=1.1, handletextpad=0.4,
               borderaxespad=0)
    fig.text(0.775, 0.916, "AT-AF3 minus GT", fontsize=6.6, va="center", color=COLORS["muted"])

    plot_x = (0.145, 0.335, 0.525)
    plot_width = 0.160
    axes = []
    for mi, (model, x) in enumerate(zip(MODELS, plot_x)):
        ax = fig.add_axes([x, 0.157, plot_width, 0.650])
        axes.append(ax)
        ax.set_ylim(ylim)
        # The original rounded rows range from 99.9 to 100.1%; no renormalization.
        ax.set_xlim(0, 100.2)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Residues (%)", labelpad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", pad=7)
        if mi == 0:
            ax.set_yticks(ypos, [r["target"] for r in records])
            ax.text(-0.64, -0.98, "All-D", transform=ax.get_yaxis_transform(),
                    fontsize=7, fontweight="bold", va="center")
            ax.text(-0.64, 8.0, "Local-D", transform=ax.get_yaxis_transform(),
                    fontsize=7, fontweight="bold", va="center")
        else:
            ax.set_yticks([])
        for i, y in enumerate(ypos):
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#F7F8FB", zorder=0)
        for tick in (0, 50, 100):
            ax.axvline(tick, color=COLORS["grid"], lw=0.45, linestyle=":", zorder=0)
        for i, (row, y) in enumerate(zip(records, ypos)):
            left = 0.0
            for c, decimal_value in zip(CLASSES, row[model]):
                value = float(decimal_value)
                if value > 0:
                    ax.barh(y, value, left=left, height=0.64, color=COLORS[c],
                            edgecolor="white", linewidth=0.35, zorder=2)
                    if value >= label_min:
                        ax.text(left + value / 2, y, f"{value:.1f}",
                                ha="center", va="center", fontsize=6.2,
                                color="white" if c == "H" else COLORS["ink"], zorder=3)
                left += value
        fig.text(x + plot_width / 2, 0.856, MODEL_LABELS[model],
                 ha="center", va="center", fontsize=7, linespacing=1.25)

    diff = differences(records)
    if np.max(np.abs(diff)) > scale:
        raise ValueError("The requested difference scale would hide measured values")
    heat = fig.add_axes([0.774, 0.157, 0.200, 0.650])
    heat.set_xlim(-0.5, 2.5)
    heat.set_ylim(ylim)
    heat.axis("off")
    cmap = LinearSegmentedColormap.from_list("muted_difference", [
        "#705894", "#BFAACB", "#E9E1EE", "#FAFBFD", "#E4EAF4", "#A7B7D4", "#3A4F80"
    ])
    norm = Normalize(-scale, scale)
    for y, values in zip(ypos, diff):
        for j, value in enumerate(values):
            background = cmap(norm(value))
            heat.add_patch(Rectangle((j - 0.48, y - 0.44), 0.96, 0.88,
                                     facecolor=background, edgecolor="none"))
            text = f"{value:+.1f}" if value != 0 else "0.0"
            heat.text(j, y, text, ha="center", va="center", fontsize=6.3,
                      color=label_color(background))
    for j, label in enumerate(CLASSES):
        fig.text(0.774 + 0.200 * (j + 0.5) / 3, 0.856,
                 label, fontsize=7, fontweight="bold", ha="center", va="center")
    color_ax = fig.add_axes([0.774, 0.097, 0.200, 0.011])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                           orientation="horizontal", ticks=[-scale, 0, scale])
    # Keep the colour scale vector-based along with the bars and heatmap.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=6.0, length=1.7, pad=2)
    colorbar.set_label("Difference (percentage points)", fontsize=6.2, labelpad=3)

    gt_dominant = np.array([np.argmax([float(v) for v in r["GT"]]) for r in records])
    at_dominant = np.array([np.argmax([float(v) for v in r["AT"]]) for r in records])
    retained = int(np.sum(gt_dominant == at_dominant))
    fig.text(0.060, 0.053,
             f"AT-AF3 retains the most abundant GT class in {retained} of {len(records)} targets.",
             fontsize=6.5, color=COLORS["ink"])
    return fig


def write_derived(records: list[dict], out_dir: Path) -> dict:
    path = out_dir / "AT_AF3_minus_GT_percentage_points.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "target", "H_delta_pp", "E_delta_pp", "C_delta_pp"])
        for row in records:
            writer.writerow([row["task"], row["target"]] +
                            [f"{row['AT'][j] - row['GT'][j]:.1f}" for j in range(3)])
    retained = sum(max(range(3), key=lambda j: row["GT"][j]) ==
                   max(range(3), key=lambda j: row["AT"][j]) for row in records)
    changed = [r["target"] for r in records if max(range(3), key=lambda j: r["GT"][j]) !=
               max(range(3), key=lambda j: r["AT"][j])]
    return {
        "source": "Target-level percentages in the user-supplied LaTeX table",
        "targets": len(records), "models": len(MODELS), "classes": len(CLASSES),
        "source_numeric_values": len(records) * len(MODELS) * len(CLASSES),
        "row_total_min": float(min(sum(r[m]) for r in records for m in MODELS)),
        "row_total_max": float(max(sum(r[m]) for r in records for m in MODELS)),
        "normalization": "Reported percentages preserved; no renormalization",
        "difference": "AT-AF3 minus GT, percentage points, using reported precision",
        "dominant_class_retained": retained, "dominant_class_changed_targets": changed,
        "max_absolute_difference_pp": float(np.max(np.abs(differences(records)))),
        "uncertainty": "No uncertainty or per-seed distribution inferred from target-level values",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("secondary_structure_source.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--difference-scale", type=float, default=30)
    parser.add_argument("--label-min", type=float, default=18.5,
                        help="Minimum segment percentage for an internal numeric label")
    args = parser.parse_args()
    if args.dpi < 300 or not np.isfinite(args.difference_scale) or args.difference_scale <= 0:
        parser.error("Use at least 300 dpi and a finite positive difference scale")
    rows = read_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = figure(rows, scale=args.difference_scale, label_min=args.label_min)
    stem = args.out_dir / "AT_AF3_secondary_structure"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(str(stem) + "." + extension, dpi=args.dpi, facecolor="white")
    fig.savefig(args.out_dir / "AT_AF3_secondary_structure_preview.png", dpi=220, facecolor="white")
    plt.close(fig)
    summary = write_derived(rows, args.out_dir)
    summary["figure_width_mm"] = 180
    summary["figure_height_mm"] = 142
    summary["png_dpi"] = args.dpi
    (args.out_dir / "figure_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
