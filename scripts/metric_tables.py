#!/usr/bin/env python3
"""One markdown table per GT-referenced metric: GT / AF3 / Corrected / deviations from GT.

Same layout as `tertiary_metric_tables.md`, extended with the metrics that live in the GT-referenced
suite, so all metrics can be quoted side by side:

  tertiary source (`tertiary_structure_summary.csv`)
    1. C-alpha RMSD to the GT           GT = 0.000 (self-superposition)
    2. radius of gyration               GT measured from the designated target chain
    3. mirror RMSD (handedness)         GT = RMSD of the GT onto its own mirror
  suite source (`gt_metrics_summary.csv`)
    4. TM-score (official USalign)      GT = 1.0000 (self-anchor measured by --self-test)
    5. lDDT (all-atom, CASP)            GT = 1.0000 (self-anchor)
    6. site-neighbourhood lDDT          GT = 1.0000 (self-anchor)
    7. coverage                         GT = 1.000 (the GT covers itself)
    8. designated-site D-chirality      GT = 1.0000 (the GT carries the designated chirality;
                                        reported per case by the metrics suite)
    9. DockQ (multi-chain)              GT = 1.000 (DockQ of the GT against itself, measured here);
                                        empty where the reference has no scorable interface

Columns: GT | AF3 | Corrected | dAF3 = |AF3-GT| | dCOR = |Corrected-GT| | dCOR-dAF3.
Model columns of the suite metrics are mean +/- SD over the 30 inference seeds; the deviation
columns use the means.  `Corrected` is C-alpha-identical to AT-AF3 by construction.

Usage:
    phenix.python scripts/metric_tables.py \
        --out reports_master42_n30_ccdfix/metric_tables.md
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tertiary_structure_rmsd as t                       # noqa: E402
from gt_metrics_suite import (DEFAULT_DOCKQ, CANON, load_chains,  # noqa: E402
                              run_dockq, target_chain_list)

REPORTS = ROOT / "reports_master42_n30_ccdfix"
CFG = t.TARGET_CFG
# tier rows: the source summary already publishes the signed delta for this metric
TIER_DELTA_KEYS = {"RMSD_AF3": "delta_RMSD"}
# (title, note, source, af3 key, corrected key, how to get the GT value)
SPECS = [
    ("C-alpha RMSD to the GT (A)",
     "Primary tertiary metric; per-chain Kabsch superposition (proper rotations only).  The GT column "
     "is 0.000 by construction, so `dAF3`/`dCOR` repeat the model values - the information is in the "
     "change column.",
     "tertiary", "RMSD_AF3", "RMSD_Corrected_ATAF3", "zero"),
    ("Radius of gyration (A)",
     "Compaction diagnostic.  `GT` is measured on the designated target chain of the experimental "
     "structure.  Positive change = the correction expanded the model (compact hairpin -> extended "
     "helix for 7P5R_2 / 8AVR_2).",
     "tertiary", "Rg_AF3", "Rg_Corrected_ATAF3", "rg"),
    ("Mirror RMSD (A) - distance to the MIRRORED GT",
     "Handedness check.  `GT` is the RMSD of the GT onto its own mirror (the cost of a perfect "
     "enantiomer mismatch).  A model value far below it means the model IS the mirror of the target; "
     "**here a positive change is the desired direction** (away from the mirror, towards the real "
     "target).  Necessary but not sufficient - see 7P5R_2 / 8AVR_2.",
     "tertiary", "RMSD_to_mirrored_GT_AF3", "RMSD_to_mirrored_GT_Corrected", "mirror"),
    ("TM-score (official USalign)",
     "Primary GT comparison of the metrics suite: per-chain mean over the scored chains; `GT` is the "
     "self-anchor 1.0000 asserted by `--self-test`.",
     "suite", "TM_mean", "TM_mean", "one"),
    ("lDDT (all-atom, CASP definition)",
     "Mean over the scored chains (pooled by pair count); `GT` is the self-anchor 1.0000.  lDDT is "
     "mirror-blind, so it never measures chirality - see the chirality table.",
     "suite", "lDDT", "lDDT", "one"),
    ("Site-neighbourhood lDDT",
     "lDDT restricted to +-4 residues around each designated site, averaged over the sites of a "
     "model; `GT` is the self-anchor 1.0000.",
     "suite", "site_lDDT", "site_lDDT", "one"),
    ("Coverage",
     "Matched GT residues / GT residues of the scored chains; `GT` = 1.000 (the GT covers itself).  "
     "Missing regions cannot raise a score - they show up here.",
     "suite", "coverage", "coverage", "one"),
    ("Designated-site D-chirality accuracy",
     "Signed-volume test at the designated sites; `GT` = 1.0000 (the reference carries the designated "
     "chirality, checked per case by the suite's integrity table).  Reported separately because "
     "distance metrics cannot see chirality.",
     "suite", "chirality_acc", "chirality_acc", "one"),
    ("DockQ (multi-chain targets)",
     "External DockQ, mean over the interfaces found.  `GT` is DockQ of the GT against itself "
     "(measured here, 1.000 when the reference has scorable interfaces).  `n/a` means no scorable "
     "interface was found between the *scored* chains (the model is a single chain there) - DockQ is "
     "undefined, not zero.  **Caveat:** this column is not used for scoring.  DockQ is run with "
     "its default `parse_hetatms=False` while every D-residue of the canonical references is a "
     "`HETATM` record, so the D chains are invisible and only the L-subset is scored.  "
     "See `dockq_interface_report.md` (`scripts/dockq_interface_eval.py`) for the per-interface "
     "values, which retain the D-residues.",
     "suite", "dockq_mean", "dockq_mean", "dockq"),
]


def num(x):
    """float(x) or nan (empty cells in the summary CSVs, e.g. DockQ without interfaces)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if v != v else f"{v:.{nd}f}"


def dev(x, nd=3):
    """A deviation cell; `n/a` when the underlying model value is undefined (e.g. DockQ)."""
    v = fmt(x, nd)
    return v if v else "n/a"


def cell(x, sd=None, nd=4):
    """`mean +/- SD` for the suite metrics, plain number for the tertiary ones."""
    v = fmt(x, nd)
    if not v:
        return "n/a"          # undefined, e.g. DockQ without a scorable interface - not zero
    return f"{v} $\\pm$ {fmt(sd, nd)}" if sd not in (None, "") else v


def gt_mirror(case: str, dataset: str) -> float:
    """RMSD of the canonical GT onto its own mirror (mean over the scored chains)."""
    kind = "All_D" if dataset == "All-D" else "Local_D"
    gt = CANON / f"{kind}_{case}_seed_0_GT.pdb"
    gt_seq, _ca = t.load(gt)
    chains = target_chain_list(case, dataset, {c: {"seq": gt_seq[c]} for c in gt_seq}, CFG)
    return t.chainwise_mirror_rmsd(gt, gt, chains)


def gt_dockq(case: str, dataset: str, workdir: Path):
    """DockQ of the canonical GT against itself, when the reference has >= 2 chains."""
    kind = "All_D" if dataset == "All-D" else "Local_D"
    gt = CANON / f"{kind}_{case}_seed_0_GT.pdb"
    if len(load_chains(gt)) < 2:
        return None
    summary, _note = run_dockq(DEFAULT_DOCKQ, gt, gt, workdir, f"gt_self_{case}")
    return summary.get("dockq_mean") if summary else None


def load_tertiary(path: Path) -> tuple:
    """({(dataset, case): row}, {tier: row}) from tertiary_structure_summary.csv."""
    all_rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    pro = {("All-D" if r["task"] == "All-D" else "Local-D", r["target"]): r
           for r in all_rows if r["scope"] == "protein"}
    ov = {(r["task"] if r["task"] in ("All-D", "Local-D") else "All 16"): r
          for r in all_rows if r["scope"] == "overall"}
    return pro, ov


def load_suite(path: Path) -> tuple:
    """({(dataset, case, condition): row}, {(dataset, condition): row}) from gt_metrics_summary.csv."""
    all_rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    pro = {(r["dataset"], r["target"], r["condition"]): r
           for r in all_rows if r["scope"] == "protein"}
    ov = {(r["dataset"], r["condition"]): r for r in all_rows if r["scope"] == "overall"}
    return pro, ov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tertiary-csv", type=Path, default=REPORTS / "tertiary_structure_summary.csv")
    ap.add_argument("--suite-csv", type=Path, default=REPORTS / "gt_metrics_summary.csv")
    ap.add_argument("--out", type=Path, default=REPORTS / "metric_tables.md")
    ap.add_argument("--no-dockq", action="store_true")
    a = ap.parse_args()

    ter, ter_all = load_tertiary(a.tertiary_csv)
    suite, suite_all = load_suite(a.suite_csv)
    cases = [("All-D", c) for c in t.ALL_D] + [("Local-D", c) for c in t.LOCAL_D]
    workdir = Path(tempfile.mkdtemp(prefix="metric_tables_"))

    gt_mir = {case: gt_mirror(case, ds) for ds, case in cases}
    gt_dq = {case: (None if a.no_dockq else gt_dockq(case, ds, workdir)) for ds, case in cases}

    def gt_of(kind: str, dataset: str, case: str):
        if kind == "zero":
            return 0.0
        if kind == "one":
            return 1.0
        if kind == "rg":
            return float(ter[(dataset, case)]["Rg_GT"])
        return gt_mir[case] if kind == "mirror" else gt_dq.get(case)

    lines = ["# GT-referenced metrics: GT / AF3 / Corrected and the deviations from the GT", "",
             "Generated by `scripts/metric_tables.py`, one table per metric.  Sources:",
             "`tertiary_structure_summary.csv` (metrics 1-3) and `gt_metrics_summary.csv` (metrics 4-9).",
             "",
             "Columns: `GT` = experimental reference, `AF3` = baseline, `Corrected` = Corrected-AT-AF3,",
             "`dAF3` = |AF3 - GT|, `dCOR` = |Corrected - GT|, `dCOR-dAF3` (positive = the correction",
             "moved *away* from the reference; the mirror table inverts this polarity, see its note).",
             "Suite metrics carry mean $\\pm$ SD over the 30 inference seeds; the deviation columns use",
             "the means.  `Corrected` is C-alpha-identical to AT-AF3 by construction.",
             "",
             "The four targets whose entry deposits both enantiomers (7P5R_2, 8AVR_2, 8BFD_2, 8P6Q_2)",
             "are scored against the **D** chain that `configs/target_chirality` designates.",
             "",
             "Tier rows reuse the overall rows of the source summaries when they exist",
             "(the `All-D` / `Local-D` rows, equal weight per protein); the `All 16` row of the suite",
             "tables is computed here because `gt_metrics_summary.csv` has no such row.  `n/a` means the",
             "quantity is undefined for that target (e.g. DockQ with no scorable interface) - never 0.",
             ""]
    for title, note, source, af3_key, cor_key, gt_kind in SPECS:
        nd = 3 if source == "tertiary" else 4
        lines += [f"## {title}", "", note, "", "| Case | GT | AF3 | Corrected | dAF3 | dCOR | dCOR-dAF3 |",
                  "|---|---|---|---|---|---|---|"]
        for tier in ("All-D", "Local-D", "All 16"):
            sel = cases if tier == "All 16" else [(d, c) for d, c in cases if d == tier]
            rows_tier = []
            for dataset, case in sel:
                if source == "tertiary":
                    row = ter[(dataset, case)]
                    av, cv = num(row[af3_key]), num(row[cor_key])
                    asd = csd = ""
                else:
                    arow = suite[(dataset, case, "AF3")]
                    av, asd = num(arow[af3_key + "_mean"]), num(arow[af3_key + "_sd"])
                    crow = suite[(dataset, case, "Corrected AT-AF3")]
                    cv, csd = num(crow[cor_key + "_mean"]), num(crow[cor_key + "_sd"])
                g = gt_of(gt_kind, dataset, case)
                rows_tier.append((case, g, av, cv, asd, csd))
            ok = [r for r in rows_tier
                  if r[1] == r[1] and r[2] == r[2] and r[3] == r[3]]
            # tier values: reuse the overall rows of the source summaries
            if source == "tertiary":
                tr, asd, csd = ter_all[tier], "", ""
                am, cm = num(tr[af3_key]), num(tr[cor_key])
                gm = num(tr["Rg_GT"]) if gt_kind == "rg" else 0.0 if gt_kind == "zero" else None
            elif tier in ("All-D", "Local-D"):
                tr = suite_all[(tier, "AF3")]
                am, asd = num(tr[af3_key + "_mean"]), num(tr[af3_key + "_sd"])
                tr = suite_all[(tier, "Corrected AT-AF3")]
                cm, csd = num(tr[cor_key + "_mean"]), num(tr[cor_key + "_sd"])
                gm = 1.0 if gt_kind == "one" else None
            else:               # the suite summary has no All-16 row: equal-weight mean over cases
                am = statistics.mean(r[2] for r in ok) if ok else float("nan")
                cm = statistics.mean(r[3] for r in ok) if ok else float("nan")
                asd = csd = ""
                gm = 1.0 if gt_kind == "one" else None
            if gm is None:      # mirror / dockq: mean of the measured per-case GT values
                gm = statistics.mean(r[1] for r in ok) if ok else float("nan")
            gtxt = cell(gm, "", nd) if source == "suite" else fmt(gm, nd)
            dkey = TIER_DELTA_KEYS.get(af3_key)
            delta = float("nan")
            if source == "tertiary" and dkey and dkey in ter_all[tier]:
                delta = num(ter_all[tier][dkey])
            if delta != delta:      # not published for this metric: difference of the tier means
                delta = abs(cm - gm) - abs(am - gm)
            lines.append(f"| **{tier}** | {gtxt} | {cell(am, asd, nd) if source == 'suite' else fmt(am, nd)} "
                         f"| {cell(cm, csd, nd) if source == 'suite' else fmt(cm, nd)} "
                         f"| {dev(abs(am - gm), nd)} | {dev(abs(cm - gm), nd)} | {dev(delta, nd)} |")
            if tier == "All 16":
                continue        # the cases were already listed under All-D / Local-D
            for case, g, av, cv, asd, csd in rows_tier:
                if g is None or g != g:
                    lines.append(f"| {case} | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                atxt = cell(av, asd, nd) if source == "suite" else fmt(av, nd)
                ctxt = cell(cv, csd, nd) if source == "suite" else fmt(cv, nd)
                lines.append(f"| {case} | {fmt(g, nd)} | {atxt} | {ctxt} | {dev(abs(av - g), nd)} "
                             f"| {dev(abs(cv - g), nd)} | {dev(abs(cv - g) - abs(av - g), nd)} |")
        lines.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"wrote {a.out} ({len(SPECS)} tables, {len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
