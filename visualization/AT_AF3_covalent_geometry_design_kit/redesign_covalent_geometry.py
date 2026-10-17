#!/usr/bin/env python3
"""Create a compact four-panel covalent-geometry figure from measured CSV data.

Preview: python redesign_covalent_geometry.py --preview --out layout_preview
Data:    python redesign_covalent_geometry.py --bonds bond_classes.csv \
             --changes correction_changes.csv --summary overall_summary.csv \
             --out covalent_geometry

The preview contains no invented measurements. Its summary numbers are
transcribed from the supplied figure; all bond-level/distribution plots are empty.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "ink": "#303645", "muted": "#727B8C", "grid": "#E3E7EF",
    "af3": "#9CAAC5", "corrected": "#354872", "lavender": "#B28DB8",
    "bond_nca": "#7588B0", "bond_co": "#B28DB8",
    "bond_cac": "#A9B8D6", "bond_cn": "#354872", "bond_cacb": "#8D93BA",
}

BONDS = [
    ("n_ca_pro", "N–Cα (PRO)"),
    ("n_ca_gly", "N–Cα (GLY)"),
    ("n_ca_other", "N–Cα (other)"),
    ("c_o_all", "C=O (all)"),
    ("ca_c_gly", "Cα–C (GLY)"),
    ("ca_c_other", "Cα–C (other)"),
    ("peptide_c_n_xpro", "C–N (X–PRO)"),
    ("peptide_c_n_nonxpro", "C–N (non-X–PRO)"),
    ("ca_cb_ala", "Cα–Cβ (ALA)"),
    ("ca_cb_iltval", "Cα–Cβ (ILE/THR/VAL)"),
    ("ca_cb_other", "Cα–Cβ (other)"),
]
TASKS = ("All-D", "Local-D")
GROUPS = [(-0.5, 2.5), (2.5, 3.5), (3.5, 5.5), (5.5, 7.5), (7.5, 10.5)]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.6, "axes.labelsize": 7.6,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.3,
        "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["ink"],
        "axes.edgecolor": COLORS["muted"], "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 7.2,
        "svg.fonttype": "none", "svg.hashsalt": "covalent-geometry-template",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [dict(r) for r in reader if any((v or "").strip() for v in r.values())]
    if not rows:
        raise ValueError(f"{path}: no measured rows supplied")
    return rows


def number(value: str | None, location: str, nonnegative: bool = True) -> float:
    try:
        v = float(value if value is not None else "")
    except ValueError as exc:
        raise ValueError(f"{location}: enter a measured numeric value") from exc
    if not np.isfinite(v) or (nonnegative and v < 0):
        raise ValueError(f"{location}: value must be finite" +
                         (" and nonnegative" if nonnegative else ""))
    return v


def read_bonds(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = csv_rows(path, {"task", "bond_class", "af3_pct", "corrected_pct"})
    expected = {(task, key) for task in TASKS for key, _ in BONDS}
    found: dict[tuple[str, str], tuple[float, float]] = {}
    for i, row in enumerate(rows, 2):
        key = (row["task"].strip(), row["bond_class"].strip())
        if key not in expected:
            raise ValueError(f"{path}:{i}: unexpected task/bond class {key}")
        if key in found:
            raise ValueError(f"{path}:{i}: duplicate task/bond class {key}")
        found[key] = (number(row["af3_pct"], f"{path}:{i}:af3_pct"),
                      number(row["corrected_pct"], f"{path}:{i}:corrected_pct"))
    if set(found) != expected:
        raise ValueError(f"{path}: missing measured rows {sorted(expected - set(found))}")
    return {
        task: tuple(np.array([found[(task, key)][j] for key, _ in BONDS])
                    for j in (0, 1))
        for task in TASKS
    }


def read_summary(path: Path) -> dict[str, tuple[float, float]]:
    rows = csv_rows(path, {"task", "af3_pct", "corrected_pct"})
    result = {}
    for i, row in enumerate(rows, 2):
        task = row["task"].strip()
        if task not in TASKS or task in result:
            raise ValueError(f"{path}:{i}: unexpected or duplicate task {task!r}")
        result[task] = (number(row["af3_pct"], f"{path}:{i}:af3_pct"),
                        number(row["corrected_pct"], f"{path}:{i}:corrected_pct"))
    if set(result) != set(TASKS):
        raise ValueError(f"{path}: both All-D and Local-D summary rows are required")
    return result


def heading(fig, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, fontsize=10.5, fontweight="bold", va="top")
    fig.text(x + 0.034, y - 0.002, title, fontsize=8.5,
             fontweight="bold", va="top")


def molecule(ax) -> None:
    """Connectivity only: no stereochemical orientation is implied."""
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    atoms = {
        "N1": (0.5, 3.1, "N"), "A1": (2.0, 3.7, "Cα"),
        "C1": (3.6, 3.1, "C"), "O1": (3.6, 4.8, "O"),
        "B1": (2.0, 1.9, "Cβ"), "N2": (5.1, 3.7, "N"),
        "A2": (6.6, 3.1, "Cα"), "C2": (8.2, 3.7, "C"),
        "O2": (8.2, 5.4, "O"), "B2": (6.6, 1.3, "Cβ"),
    }
    connections = [
        ("N1", "A1", "bond_nca"), ("A1", "C1", "bond_cac"),
        ("C1", "N2", "bond_cn"), ("N2", "A2", "bond_nca"),
        ("A2", "C2", "bond_cac"), ("A1", "B1", "bond_cacb"),
        ("A2", "B2", "bond_cacb"),
    ]
    for a, b, color in connections:
        ax.plot([atoms[a][0], atoms[b][0]], [atoms[a][1], atoms[b][1]],
                color=COLORS[color], lw=2.2, solid_capstyle="round", zorder=1)
    for c, o in (("C1", "O1"), ("C2", "O2")):
        for offset in (-0.065, 0.065):
            ax.plot([atoms[c][0] + offset, atoms[o][0] + offset],
                    [atoms[c][1], atoms[o][1]], color=COLORS["bond_co"], lw=1.4)
    for x, y, label in atoms.values():
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
    ax.text(2.0, 0.35, "Residue i", ha="center", color=COLORS["muted"], fontsize=6.8)
    ax.text(6.6, 0.35, "Residue i+1", ha="center", color=COLORS["muted"], fontsize=6.8)


def comparison(ax, data, *, show_labels: bool, xmax: float, preview: bool) -> None:
    for i, (lo, hi) in enumerate(GROUPS):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#F5F6FA", zorder=0)
        if i:
            ax.axhline(lo, color=COLORS["grid"], lw=0.5, zorder=0)
    # Small negative left margin so markers near zero are not clipped.
    ax.set(xlim=(-0.045 * xmax, xmax), ylim=(10.6, -0.6))
    ax.set_xticks(np.arange(0, xmax + 1e-9, 1.0 if xmax <= 5 else 2.0))
    ax.set_yticks(np.arange(len(BONDS)))
    ax.set_yticklabels([label for _, label in BONDS] if show_labels else [])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=7)
    ax.grid(axis="x", color=COLORS["grid"], linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Normalized deviation from GT (%)", labelpad=5)
    if preview:
        ax.text(0.5, 0.52, "Measured bond-class\nvalues go here", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["muted"], fontsize=7.7,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
        return
    af3, corrected = data
    y = np.arange(len(BONDS))
    ax.hlines(y, np.minimum(af3, corrected), np.maximum(af3, corrected),
              color="#C9D0DE", lw=1.2, zorder=2)
    ax.scatter(af3, y, s=22, facecolors="white", edgecolors=COLORS["af3"],
               linewidths=1.0, zorder=3)
    ax.scatter(corrected, y, s=15, c=COLORS["corrected"],
               edgecolors="white", linewidths=0.2, zorder=4)


def correction_distribution(ax, changes: np.ndarray | None) -> None:
    ax.set_ylabel("Cumulative\npredictions (%)", labelpad=4)
    ax.set_xlabel("Normalized bond-length change (%)", labelpad=4)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linestyle=":", color=COLORS["grid"], linewidth=0.5)
    if changes is None:
        ax.set_xlim(0, 0.016)
        ax.set_xticks([0, 0.004, 0.008, 0.012, 0.016])
        ax.text(0.5, 0.53, "Plot the measured distribution\nwithout overlapping zero values",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.1, color=COLORS["muted"])
        return
    values, counts = np.unique(changes, return_counts=True)
    cumulative = np.cumsum(counts) * 100 / counts.sum()
    lo = min(0.0, float(values[0]))
    span = max(float(values[-1]) - lo, 0.001)
    hi = float(values[-1]) + 0.035 * span
    # The leading point preserves the full ECDF jump at zero, including ties.
    ax.step(np.r_[lo, values, hi], np.r_[0.0, cumulative, 100.0],
            where="post", color=COLORS["corrected"], lw=1.3)
    ax.set_xlim(lo - 0.015 * span, hi)
    zero_fraction = np.count_nonzero(changes == 0) / len(changes)
    ax.text(0.98, 0.09, f"Zero change: {zero_fraction:.1%}", transform=ax.transAxes,
            ha="right", fontsize=6.8, color=COLORS["muted"])


def make_figure(bonds=None, changes=None, summaries=None, *, preview=False, xmax=4.0):
    style()
    fig = plt.figure(figsize=(7.1, 5.45), facecolor="white")
    heading(fig, 0.035, 0.955, "a", "11 bond classes in five families")
    heading(fig, 0.625, 0.955, "b", "Change during correction")
    # fig.text(0.659, 0.907, "AT-AF3 → Corrected AT-AF3", fontsize=7.0,
    #          color=COLORS["muted"])
    mol = fig.add_axes([0.055, 0.732, 0.50, 0.176])
    molecule(mol)
    families = [
        ("N–Cα · 3", "bond_nca"), ("C=O · 1", "bond_co"),
        ("Cα–C · 2", "bond_cac"), ("C–N · 2", "bond_cn"),
        ("Cα–Cβ · 3", "bond_cacb"),
    ]
    for i, (label, color) in enumerate(families):
        x = 0.068 + i * 0.102
        fig.add_artist(Line2D([x, x + 0.014], [0.702, 0.702],
                             transform=fig.transFigure, color=COLORS[color], lw=2))
        fig.text(x + 0.019, 0.702, label, va="center", fontsize=6.5)
    # fig.text(0.058, 0.669, "Connectivity schematic; counts indicate residue-context subclasses.",
    #          fontsize=6.5, color=COLORS["muted"])
    ecdf = fig.add_axes([0.707, 0.728, 0.255, 0.150])
    correction_distribution(ecdf, changes)
    if preview:
        note = "Source figure: mean 0.00049%; median 0.00000%"
    else:
        note = f"n = {len(changes)}; mean {np.mean(changes):.5f}%; median {np.median(changes):.5f}%"
    # fig.text(0.653, 0.648, note, fontsize=6.6, color=COLORS["muted"])

    heading(fig, 0.035, 0.592, "c", "All-D")
    heading(fig, 0.625, 0.592, "d", "Local-D")
    handles = [
        Line2D([], [], marker="o", ms=4.5, linestyle="none", markerfacecolor="white",
               markeredgecolor=COLORS["af3"], label="AF3"),
        Line2D([], [], marker="o", ms=4.2, linestyle="none", markerfacecolor=COLORS["corrected"],
               markeredgecolor=COLORS["corrected"], label="AT-AF3"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.355, 0.596),
               ncol=2, columnspacing=1.4, handletextpad=0.4)
    left = fig.add_axes([0.257, 0.135, 0.280, 0.408])
    right = fig.add_axes([0.680, 0.135, 0.282, 0.408])
    comparison(left, None if preview else bonds["All-D"], show_labels=True,
               xmax=xmax, preview=preview)
    comparison(right, None if preview else bonds["Local-D"], show_labels=False,
               xmax=xmax, preview=preview)
    if summaries:
        for task, center in (("All-D", 0.39), ("Local-D", 0.816)):
            af3, corrected = summaries[task]
            # fig.text(center, 0.047, f"Overall mean: AF3 {af3:.4f}% · Corrected {corrected:.4f}%",
            #          ha="center", fontsize=6.8, color=COLORS["muted"])
    if preview:
        fig.text(0.5, 0.015,
                 "LAYOUT PREVIEW · Summary labels copied from the supplied figure; measured plots left empty.",
                 ha="center", fontsize=6.1, color=COLORS["muted"])
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="Create an explicitly labelled layout, with no invented data.")
    parser.add_argument("--bonds", type=Path)
    parser.add_argument("--changes", type=Path)
    parser.add_argument("--summary", type=Path, help="Optional pooled summaries, computed with the original weighting.")
    parser.add_argument("--out", type=Path, default=Path("covalent_geometry"))
    parser.add_argument("--xmax", type=float, help="Common upper limit for panels c/d; clipping is rejected.")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    try:
        if args.preview:
            if args.bonds or args.changes or args.summary:
                parser.error("Use --preview alone; measured input belongs in data mode.")
            bonds, changes = None, None
            summaries = {"All-D": (0.0406, 0.9124), "Local-D": (0.1779, 0.5488)}
            xmax = args.xmax if args.xmax is not None else 4.0
        else:
            if args.bonds is None or args.changes is None:
                parser.error("Supply both --bonds and --changes, or use --preview.")
            bonds = read_bonds(args.bonds)
            rows = csv_rows(args.changes, {"change_pct"})
            # Preserve the supplied metric, including its sign; do not take abs().
            changes = np.array([number(row["change_pct"], f"{args.changes}:{i}:change_pct", False)
                                for i, row in enumerate(rows, 2)])
            summaries = read_summary(args.summary) if args.summary else None
            maximum = max(float(arr.max()) for pair in bonds.values() for arr in pair)
            xmax = args.xmax if args.xmax is not None else max(4.0, np.ceil(maximum * 1.05))
            if maximum > xmax:
                raise ValueError("--xmax would hide measured points; choose a larger limit")
        if not np.isfinite(xmax) or xmax <= 0:
            raise ValueError("--xmax must be finite and positive")
        fig = make_figure(bonds, changes, summaries, preview=args.preview, xmax=xmax)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("svg", "png"):
            destination = Path(str(args.out) + "." + extension)
            fig.savefig(destination, dpi=args.dpi, facecolor="white")
            print(destination)
        plt.close(fig)
    except (ValueError, OSError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
