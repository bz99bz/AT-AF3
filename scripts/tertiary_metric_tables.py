#!/usr/bin/env python3
"""One markdown table per tertiary-structure metric: GT / AF3 / Corrected / deviations from GT.

Metrics
  1. C-alpha RMSD to the GT (primary; chains paired by sequence, proper superposition only)
  2. radius of gyration (compaction)
  3. mirror RMSD against the *mirrored* GT (handedness / enantiomer check)

The model columns come from `tertiary_structure_summary.csv`; the GT column of the mirror metric is
measured here with the same functions (`tertiary_structure_rmsd`), so the numbers are consistent with
the report.  The four targets whose entry deposits both enantiomers are scored against the D chain
(see `docs/METRICS.md` for the convention).

Usage:
    phenix.python scripts/tertiary_metric_tables.py \
        --from-csv reports_master42_n30_ccdfix/tertiary_structure_summary.csv \
        --out reports_master42_n30_ccdfix/tertiary_metric_tables.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tertiary_structure_rmsd as t            # noqa: E402  (load, chainwise_mirror_rmsd)
from gt_metrics_suite import target_chain_list  # noqa: E402

CANON = t.CANON
CFG = t.TARGET_CFG
METRICS = [
    ("C-alpha RMSD to the GT (A)",
     "Primary metric: per-chain Kabsch superposition (rotation + translation only, never a "
     "reflection), averaged over the scored chains.  The GT column is 0.000 by construction (the GT "
     "superposed onto itself), so the two deviation columns repeat the model values; the informative "
     "summary is the last column, `dCOR - dAF3`.",
     "RMSD_AF3", "RMSD_Corrected_ATAF3"),
    ("Radius of gyration (A)",
     "Compaction diagnostic: mean over the scored chains of the C-alpha radius of gyration.  `GT` is "
     "taken from the designated target chain of the experimental structure.  A large positive "
     "`dCOR - dAF3` means the corrected model expanded relative to AF3 (the compact-hairpin to "
     "extended-helix failure).",
     "Rg_AF3", "Rg_Corrected_ATAF3"),
    ("Mirror RMSD (A) - distance to the MIRRORED GT",
     "Handedness check.  `GT` is the RMSD of the experimental structure onto its own mirror, i.e. what "
     "a perfect enantiomer mismatch costs for that target; the model columns are the model's RMSD to "
     "the mirrored GT.  A model value far below the GT column means the model IS the mirror of the "
     "target (AF3 is L-only, so this is expected for all-D targets); a value close to the GT column "
     "means the model is chirally unrelated to it.  **In this table a positive `dCOR-dAF3` is the "
     "desired direction** (the correction moved the model away from the mirror, i.e. towards the real "
     "target) - the opposite polarity of the two tables above, where the GT column is the reference "
     "itself.  Read it together with the first table: moving away from the mirror is necessary but not "
     "sufficient - 7P5R_2 and 8AVR_2 do move away from the mirror (0.46 -> 13.05 and 0.34 -> 13.14 A) "
     "yet remain 13.6 / 13.7 A from the real target, i.e. they left the enantiomer without finding "
     "the fold.",
     "RMSD_to_mirrored_GT_AF3", "RMSD_to_mirrored_GT_Corrected"),
]


def fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{nd}f}"


def gt_mirror(case: str, dataset: str) -> float:
    """RMSD of the canonical GT onto its own mirror (mean over the scored chains)."""
    kind = "All_D" if dataset == "All-D" else "Local_D"
    gt = CANON / f"{kind}_{case}_seed_0_GT.pdb"
    gt_seq, _ca = t.load(gt)
    tchains = target_chain_list(case, dataset, {c: {"seq": gt_seq[c]} for c in gt_seq}, CFG)
    return t.chainwise_mirror_rmsd(gt, gt, tchains)


def gt_value(title: str, row: dict, rows: list, gt_mir: dict):
    """The GT reference value for one metric and one row (C-alpha: 0.0 = GT superposed on itself)."""
    if title.startswith("C-alpha"):
        return 0.0
    if title.startswith("Radius"):
        return float(row["Rg_GT"])          # populated for the tier rows too (mean over proteins)
    if row["scope"] == "protein":
        return gt_mir.get((row["task"], row["target"]), float("nan"))
    want = (row["task"],) if row["task"] in ("All-D", "Local-D") else ("All-D", "Local-D")
    vals = [gt_mir[(r["task"], r["target"])] for r in rows
            if r["scope"] == "protein" and r["task"] in want
            and (r["task"], r["target"]) in gt_mir]
    return statistics.mean(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-csv", type=Path,
                    default=ROOT / "reports_master42_n30_ccdfix/tertiary_structure_summary.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports_master42_n30_ccdfix/tertiary_metric_tables.md")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(a.from_csv.open(newline="", encoding="utf-8"))
            if r["scope"] in ("protein", "overall")]
    gt_mir = {(r["task"], r["target"]): gt_mirror(r["target"], r["task"])
              for r in rows if r["scope"] == "protein"}

    lines = ["# Tertiary-structure metrics: GT / AF3 / Corrected and the deviations from the GT",
             "",
             "Generated by `scripts/tertiary_metric_tables.py` from `tertiary_structure_summary.csv`",
             "(30 inference seeds per target).  One table per metric; the three metrics of the",
             "tertiary-structure analysis are **C-alpha RMSD (primary)**, **radius of gyration",
             "(compaction)** and **mirror RMSD (handedness)**.",
             "",
             "Columns: `GT` = experimental reference, `AF3` = baseline prediction, `Corrected` =",
             "Corrected-AT-AF3, `dAF3` = |AF3 - GT|, `dCOR` = |Corrected - GT|, `dCOR-dAF3` = the change",
             "(positive = the correction moved *away* from the GT).  `Corrected` is C-alpha-identical to",
             "AT-AF3 by construction.",
             "",
             "The four targets whose deposited entry contains both enantiomers (7P5R_2, 8AVR_2,",
             "8BFD_2, 8P6Q_2) are scored against the **D** chain that `configs/target_chirality`",
             "designates - see `docs/METRICS.md` for the convention.",
             ""]
    for title, note, af3_key, cor_key in METRICS:
        lines += [f"## {title}", "", note, "",
                  "| Case | GT | AF3 | Corrected | dAF3 | dCOR | dCOR-dAF3 |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            gv = gt_value(title, r, rows, gt_mir)
            af3, cor = float(r[af3_key]), float(r[cor_key])
            d_a, d_c = abs(af3 - gv), abs(cor - gv)
            label = r["target"] if r["scope"] == "protein" else f"**{r['target']}**"
            lines.append(f"| {label} | {fmt(gv)} | {fmt(af3)} | {fmt(cor)} | {fmt(d_a)} | "
                         f"{fmt(d_c)} | {fmt(d_c - d_a)} |")
        lines.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

