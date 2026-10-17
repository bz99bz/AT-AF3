#!/usr/bin/env python3
"""Figure 2 - chirality heatmap and paired effects.

Panel a : 16 x 3 heatmap (7 All-D + 9 Local-D proteins x AF3 / AT-AF3 / Corrected)
Panel b : All-D paired AF3 -> AT-AF3 accuracy plot
Panel c : Local-D paired AF3 -> AT-AF3 -> Corrected accuracy plot

Numbers are read from the corrected-CCD benchmark, not transcribed by hand:
  All-D   : reports_master42_n30_ccdfix/seed_level_metrics.csv  (mean over 30 seeds)
  Local-D : computed from runs/local_d_curated_v1_1_post_cutoff (evaluable sites
            only), cached to reports_master42_n30_ccdfix/locald_case_condition_accuracy.csv

Run with an interpreter that has gemmi and matplotlib installed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ALL_D_CASES = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
LOCAL_D_CASES = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1",
                 "8U78_1", "8UM9_1", "9BIC_1"]
ALLD_COND = {"AF3": "standard_af3", "AT-AF3": "at_af3",
             "Corrected": "at_af3_chirality_corrected"}
LOCALD_COND = {"AF3": "AF3", "AT-AF3": "AT-AF3", "Corrected": "Corrected ATAF3"}
CONDS = ["AF3", "AT-AF3", "Corrected"]


def frame(cases, values):
    rows = {"Protein": cases}
    for i, cond in enumerate(CONDS):
        rows[cond] = [values[c][i] for c in cases]
    return pd.DataFrame(rows)


def load_all_d():
    path = ROOT / "reports_master42_n30_ccdfix/seed_level_metrics.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run the All-D pipeline before plotting Figure 2."
        )
    df = pd.read_csv(path)
    vals = {}
    for case in ALL_D_CASES:
        vals[case] = [float(df[(df.case_id == case) & (df.condition == ALLD_COND[c])]
                            ["target_D_accuracy"].mean()) for c in CONDS]
    return frame(ALL_D_CASES, vals)


def load_local_d():
    cache = ROOT / "reports_master42_n30_ccdfix/locald_case_condition_accuracy.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        vals = {r["case_id"]: [float(r[c]) for c in CONDS] for _, r in df.iterrows()}
        return frame(LOCAL_D_CASES, vals)
    try:
        import locald_evaluable_table as L
    except Exception as exc:
        raise SystemExit(
            "Local-D recomputation unavailable (%s). Run the Local-D pipeline "
            "before plotting Figure 2." % exc
        ) from exc

    tm = ROOT / "data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv"
    targets = {}
    for _, r in pd.read_csv(tm).iterrows():
        if r["case_id"] not in L.CASES or str(r.get("target_chirality", "D")).upper() != "D":
            continue
        raw = r.get("label_seq_id") or r.get("residue_id")
        # The manifest's chain letters are source-CIF based; the run outputs may use different
        # ones (7W40_6's DPN is chain E in the reference entry but chain F in the models), so
        # resolve the chain through the CCD code - same rule as locald_evaluable_table.py.
        chain = r["chain_id"]
        hits = L.ccd_chain(r["case_id"], str(r["ccd_id"]).strip().upper())
        if len(hits) == 1 and hits[0] != chain:
            print(f"  note: {r['case_id']} {r['ccd_id']} is on canonical chain {hits[0]} "
                  f"(manifest says {chain}) - using the CCD chain")
            chain = hits[0]
        try:
            targets.setdefault(r["case_id"], {}).setdefault(chain, set()).add(int(raw))
        except (TypeError, ValueError):
            pass

    seeds = sorted({p.name.split("_")[1] for p in (L.ROOTDIR / "standard_af3" / L.CASES[0]).glob("seed_*")},
                   key=int)
    vals, cache_rows = {}, []
    for case in LOCAL_D_CASES:
        vals[case] = []
        for label in CONDS:
            scores = []
            for seed in seeds:
                p = L.find_model(LOCALD_COND[label], case, seed)
                if p is None:
                    continue
                n, k = L.score(p, targets.get(case, {}))
                if n:
                    scores.append(k / n)
            vals[case].append(float(np.mean(scores)) if scores else 0.0)
        cache_rows.append({"case_id": case, **{c: v for c, v in zip(CONDS, vals[case])}})
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_rows).to_csv(cache, index=False)
    print("cached Local-D case/condition accuracy ->", cache)
    return frame(LOCAL_D_CASES, vals)


# ============================================================
# 1. Input data
# ============================================================

all_d = load_all_d()
local_d = load_local_d()

print("All-D accuracy table:\n", all_d.to_string(index=False))
print("Local-D accuracy table:\n", local_d.to_string(index=False))

# Combine datasets for heatmap
combined = pd.concat([all_d, local_d], ignore_index=True)

heatmap_data = combined[["AF3", "AT-AF3", "Corrected"]].to_numpy()


# ============================================================
# 2. Global plotting parameters
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.7,
    "figure.dpi": 150,
    "savefig.dpi": 600,

    # Keep text editable in SVG
    "svg.fonttype": "none",
})


# ============================================================
# 3. Figure layout
# ============================================================

fig = plt.figure(figsize=(10.7, 8.5))

gs = fig.add_gridspec(
    2,
    2,

    # Heatmap is taller than lower panels
    height_ratios=[1.48, 1.0],

    # Margins
    left=0.13,
    right=0.95,
    bottom=0.13,
    top=0.95,

    # Panel spacing
    wspace=0.30,
    hspace=0.38,
)


# ============================================================
# 4. Panel a: 16 x 3 heatmap
# ============================================================

ax_a = fig.add_subplot(gs[0, :])

im = ax_a.imshow(
    heatmap_data,
    vmin=0,
    vmax=1,
    aspect="auto",
)


# X axis
ax_a.set_xticks(
    [0, 1, 2],
    [
        "AF3",
        "AT-AF3",
        "AT-AF3\nCorrected",
    ],
)

ax_a.tick_params(
    axis="x",
    top=True,
    labeltop=True,
    bottom=False,
    labelbottom=False,
    length=0,
    pad=6,
)


# Y axis
ax_a.set_yticks(np.arange(len(combined)), combined["Protein"])

ax_a.tick_params(axis="y", length=0)


# ------------------------------------------------------------
# Add numerical values inside heatmap cells
# ------------------------------------------------------------

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data[i, j]
        ax_a.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7.3,

            # Make corrected values visually stronger
            fontweight=("bold" if j == 2 else "normal"),
        )


# ------------------------------------------------------------
# Separate All-D and Local-D groups
# ------------------------------------------------------------

boundary = len(all_d) - 0.5

ax_a.axhline(boundary, linewidth=1.1)


# Group labels on left
ax_a.text(
    -0.10,
    (len(all_d) - 1) / 2,
    "All-D",
    transform=(ax_a.get_yaxis_transform()),
    ha="right",
    va="center",
    rotation=90,
    fontweight="bold",
)

ax_a.text(
    -0.10,
    len(all_d) + (len(local_d) - 1) / 2,
    "Local-D",
    transform=(ax_a.get_yaxis_transform()),
    ha="right",
    va="center",
    rotation=90,
    fontweight="bold",
)


# Remove heatmap frame
for spine in ax_a.spines.values():
    spine.set_visible(False)


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(im, ax=ax_a, fraction=0.021, pad=0.018)

cbar.set_label("Target D-chirality accuracy")

cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.00])


# Panel title
ax_a.set_title(
    "a   Target-level response across baseline, "
    "stereochemical conditioning, and geometric correction",
    loc="left",
    fontweight="bold",
    pad=10,
)


# ============================================================
# 5. Panel b: All-D paired improvement plot
# ============================================================

ax_b = fig.add_subplot(gs[1, 0])

for _, row in all_d.iterrows():
    ax_b.plot(
        [0, 1, 2],
        [row["AF3"], row["AT-AF3"], row["Corrected"]],
        marker="o",
        linewidth=1.45,
        markersize=4.4,
        alpha=0.90,
        label=row["Protein"],
    )


# Axis limits
ax_b.set_xlim(-0.16, 2.16)

ax_b.set_ylim(-0.03, 1.03)


# Axis labels
ax_b.set_xticks([0, 1, 2], ["AF3", "AT-AF3", "Corrected"])

ax_b.set_ylabel("Target D-chirality accuracy")


# Horizontal guide lines
ax_b.grid(axis="y", linewidth=0.6, alpha=0.25)


# Title
ax_b.set_title(
    "b   All-D: conditioning redirects AF3; "
    "correction is sterically limited",
    loc="left",
    fontweight="bold",
    pad=8,
)


# Remove unnecessary borders
ax_b.spines["top"].set_visible(False)

ax_b.spines["right"].set_visible(False)


# Legend below panel
ax_b.legend(
    frameon=False,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    columnspacing=1.1,
    handlelength=1.7,
)


# ============================================================
# 6. Panel c: Local-D progressive paired plot
# ============================================================

ax_c = fig.add_subplot(gs[1, 1])

for _, row in local_d.iterrows():
    ax_c.plot(
        [0, 1, 2],
        [row["AF3"], row["AT-AF3"], row["Corrected"]],
        marker="o",
        linewidth=1.35,
        markersize=4.2,
        alpha=0.90,
        label=row["Protein"],
    )


# Axis limits
ax_c.set_xlim(-0.16, 2.16)

ax_c.set_ylim(-0.03, 1.03)


# Axis labels
ax_c.set_xticks([0, 1, 2], ["AF3", "AT-AF3", "Corrected"])

ax_c.set_ylabel("Target D-chirality accuracy")


# Horizontal guide lines
ax_c.grid(axis="y", linewidth=0.6, alpha=0.25)


# Title
ax_c.set_title(
    "c   Local-D: conditioning is target-dependent; "
    "correction resolves residual errors",
    loc="left",
    fontweight="bold",
    pad=8,
)


# Remove unnecessary borders
ax_c.spines["top"].set_visible(False)

ax_c.spines["right"].set_visible(False)


# Legend below panel
ax_c.legend(
    frameon=False,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    columnspacing=0.85,
    handlelength=1.6,
)


# ============================================================
# 7. Save figure
# ============================================================

png_path = ROOT / "Fig2_chirality_heatmap_paired_effects.png"
svg_path = ROOT / "Fig2_chirality_heatmap_paired_effects.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

print("wrote", png_path)
print("wrote", svg_path)
