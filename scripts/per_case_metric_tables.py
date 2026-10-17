#!/usr/bin/env python3
"""Per-case metric tables: one markdown table per metric, columns = GT / AF3 / Corrected AT-AF3.

Reads the finished `gt_metrics_per_seed.csv` (no structure re-scoring for the three prediction
columns) and **measures** the GT column by scoring each canonical GT against itself with the very
same functions the suite uses, so the GT column is a real anchor rather than an assumption
(TM = lDDT = 1, CA-RMSD = 0, coverage = 1 by construction; mirTM / chirality / site-lDDT and the
DockQ(GT, GT) self-check come from actual runs).

AT-AF3 is included as a fourth, clearly marked column because it is the
scientifically informative condition (the correction acts on top of it).

Usage:
    phenix.python scripts/per_case_metric_tables.py \
        --from-csv reports_master42_n30_ccdfix/gt_metrics_per_seed.csv \
        --out reports_master42_n30_ccdfix/per_case_metric_tables.md
"""
from __future__ import annotations

import argparse
import collections
import csv
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_metrics_suite as g   # reuse loaders, scoring, USalign and DockQ wrappers

ROOT = Path(__file__).resolve().parents[1]
# requested columns first, then the extra informative one
CONDITIONS = ["AF3", "Corrected AT-AF3", "AT-AF3"]
COND_LABEL = {"AF3": "AF3", "Corrected AT-AF3": "Corrected AT-AF3", "AT-AF3": "AT-AF3 (extra)"}
TIERS = [("All-D", None), ("Local-D", None)]


def mean_sd(rows, key):
    """(mean, sd, n) over the non-empty values of one per-seed column."""
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    if not vals:
        return None
    return (statistics.mean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def cell(value, kind="mean_sd", digits=4):
    """Format one table cell."""
    if value is None:
        return "n/a"
    if kind == "mean":
        return f"{value[0]:.{digits}f}"
    if kind == "count":
        return f"{value[0]:.1f}"
    return f"{value[0]:.{digits}f} $\\pm$ {value[1]:.{digits}f}"


def _nn(x):
    """None instead of NaN/empty."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def gt_anchor(case, dataset, canonical_dir, cfg_root, manifest, window, radius,
              usalign, dockq, workdir):
    """Measure the GT self-anchor row for one case (same functions as the suite)."""
    kind = "All_D" if dataset == "All-D" else "Local_D"
    gt_path = canonical_dir / f"{kind}_{case}_seed_0_GT.pdb"
    gt = g.load_chains(gt_path)
    sites, _notes = g.target_sites(case, dataset, gt, cfg_root, manifest)
    tchains = g.target_chain_list(case, dataset, gt, cfg_root)
    res = g.score_model(gt, gt, None, sites, window, radius, usalign=usalign,
                        tm_impl="both", workdir=workdir, tag=f"GT_{case}",
                        target_chains=tchains)
    out = {
        "TM": (res["tm_mean"], 0.0, 1),
        "mirTM": (res["mir_tm_mean"], 0.0, 1),
        "lDDT": (res["lddt"], 0.0, 1),
        "CA_RMSD": (res["ca_rmsd_mean"], 0.0, 1),
        "coverage": (res["coverage"], 0.0, 1),
        "matched": (res["n_matched_res"], 0.0, 1),
        "gt_res": res["n_gt_res"],
        "chirality": (_nn(res["chirality_acc"]), 0.0, 1),
        "sites": res["chirality_n"],
        "site_lDDT": (_nn(res["site_lddt_mean"]), 0.0, 1),
        "dockq": None, "dockq_n": "", "mirror_models": 0, "n_seeds": 1,
        "n_chains": len(gt), "n_gt_chains": len(gt), "n_scored_chains": len(gt),
    }
    # data-integrity check: does the GT itself carry the designated chirality?
    ok = wrong = none = 0
    for _label, chain, key, expected in sites:
        sign = g.chirality_sign(gt.get(chain, {}).get("xyz", {}).get(key, {}))
        if sign is None:
            none += 1
        elif sign == expected:
            ok += 1
        else:
            wrong += 1
    out.update({"gt_designated": len(sites), "gt_ok": ok, "gt_wrong": wrong,
                "gt_none": none, "designated_chain": (sites[0][1] if sites else "")})
    if dockq and len(gt) > 1:
        summary, _note = g.run_dockq(dockq, gt_path, gt_path, workdir, f"dqGT_{case}")
        if summary:
            out["dockq"] = (summary["dockq_mean"], 0.0, 1)
            out["dockq_n"] = summary["n_interfaces"]
    return out


def cond_values(rows):
    """Per-condition metric values from the per-seed rows of one (case, condition)."""
    if not rows:
        return None
    return {
        "TM": mean_sd(rows, "TM_mean"),
        "mirTM": mean_sd(rows, "mirTM_mean"),
        "lDDT": mean_sd(rows, "lDDT"),
        "CA_RMSD": mean_sd(rows, "CA_RMSD"),
        "coverage": mean_sd(rows, "coverage"),
        "matched": mean_sd(rows, "n_matched_res"),
        "gt_res": rows[0].get("n_gt_res", ""),
        "chirality": mean_sd(rows, "chirality_acc"),
        "sites": max(int(r["chirality_n"]) for r in rows
                     if str(r.get("chirality_n", "")).isdigit()),
        "site_lDDT": mean_sd(rows, "site_lDDT"),
        "dockq": mean_sd(rows, "dockq_mean"),
        "dockq_n": (max(int(r["n_interfaces"]) for r in rows
                        if str(r.get("n_interfaces", "")).isdigit())
                    if any(str(r.get("n_interfaces", "")).isdigit() for r in rows) else ""),
        "mirror_models": sum(1 for r in rows if r["enantiomer"] == "mirror"),
        "n_seeds": len(rows),
        "n_chains": mean_sd(rows, "n_chains"),
        "n_gt_chains": (max(int(r["n_gt_chains"]) for r in rows
                            if str(r.get("n_gt_chains", "")).isdigit())
                        if any(str(r.get("n_gt_chains", "")).isdigit() for r in rows) else ""),
        "n_scored_chains": (max(int(r["n_scored_chains"]) for r in rows
                                if str(r.get("n_scored_chains", "")).isdigit())
                            if any(str(r.get("n_scored_chains", "")).isdigit()
                                   for r in rows) else ""),
    }


METRICS = [
    ("TM-score (official USalign, per chain)", "TM",
     "Per-chain TM-score, mean $\\pm$ SD over the 30 seeds.  The GT column is the self-anchor."),
    ("mirTM (mirrored-GT reference)", "mirTM",
     "The same TM-score against the mirrored GT.  `mirTM > TM` means the model is the enantiomer; "
     "for the GT itself this quantifies the enantiomer sensitivity of the method."),
    ("lDDT (all-atom, CASP definition)", "lDDT",
     "Distance-difference test on reference pairs within 15 A, thresholds 0.5/1/2/4 A.  "
     "Mirror-blind by construction: an exact enantiomer of the GT scores 1.0000."),
    ("C-alpha RMSD (A)", "CA_RMSD",
     "Supplementary metric: per-chain proper superposition, averaged over the scored chains."),
    ("coverage", "coverage",
     "Matched GT residues divided by the GT residues of the **scored chains** - see the "
     "scored-chain table at the end.  A value of 1.000 does not mean the whole complex is covered."),
    ("matched GT residues (count)", "matched",
     "How many GT residues were actually compared, after the shared-scope intersection across "
     "the conditions."),
    ("designated-site D-chirality accuracy", "chirality",
     "Signed-volume test `(N-CA)x(C-CA).(CB-CA) < 0` at the designated target sites only.  "
     "The GT column is 1.0000 by construction (the sites are designated D)."),
    ("site-neighbourhood lDDT", "site_lDDT",
     "lDDT restricted to the residues within $\\pm$4 sequence positions of each designated site, "
     "averaged over the sites of a model."),
    ("DockQ (external tool, mean over interfaces)", "dockq",
     "Only computed when both the prediction and the GT have more than one chain.  An empty cell "
     "means DockQ found no scorable interface - it is **not** a score of zero.  **DockQ caveat:** "
     "the canonical references write every D-residue as `HETATM`, which DockQ's "
     "default parser skips, so these values are computed against the L-subset of each reference.  "
     "Quote `dockq_interface_report.md` instead: every "
     "D-residue retained, every reference interface scored separately with an explicit `--mapping`."),
]


def gt_check_table(order, anchors, flags):
    """Does the canonical GT itself carry the chirality the config designates?"""
    lines = ["## GT data-integrity check (designated chirality)", "",
             "Does the **canonical GT itself** carry the chirality that `configs/target_chirality` "
             "designates?  `GT reads L` counts designated sites whose signed volume says L, i.e. "
             "reference positions that are *not* the designated form.", "",
             "| Case | designated chain | designated sites | GT reads D | GT reads L | non-evaluable | flag |",
             "|---|---|---|---|---|---|---|"]
    for dataset, case in order:
        a = anchors.get((dataset, case)) or {}
        lines.append(f"| {case} | {a.get('designated_chain', '')} | {a.get('gt_designated', '')} | "
                     f"{a.get('gt_ok', '')} | {a.get('gt_wrong', '')} | {a.get('gt_none', '')} | "
                     f"{'&#9888; flagged' if (dataset, case) in flags else 'ok'} |")
    lines.append("")
    return lines


def case_table(title, key, note, order, data, anchors, flags=()):
    cols = [COND_LABEL[c] for c in CONDITIONS]
    lines = [f"## {title}", "", note, "",
             "| Case | GT | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for dataset, case in order:
        a = anchors.get((dataset, case))
        tier = "All-D" if dataset == "All-D" else "Local-D"
        mark = " &#9888;" if (dataset, case) in flags else ""
        cells = [cell(a.get(key) if a else None)]
        for c in CONDITIONS:
            v = data.get((dataset, case), {}).get(c)
            cells.append(cell(v.get(key) if v else None))
        lines.append(f"| {case} ({tier}){mark} | " + " | ".join(cells) + " |")
    for label, sel in (("**All-D (7 proteins)**", [k for k in order if k[0] == "All-D"]),
                       ("**Local-D (9 proteins)**", [k for k in order if k[0] == "Local-D"]),
                       ("**All 16 proteins**", list(order))):
        gt_vals = [anchors[k][key] for k in sel if anchors.get(k) and anchors[k].get(key)]
        cells = [cell((statistics.mean(v[0] for v in gt_vals), 0.0, 1)) if gt_vals else "n/a"]
        for c in CONDITIONS:
            vals = [data[k][c][key] for k in sel
                    if data.get(k, {}).get(c) and data[k][c].get(key)]
            cells.append(cell((statistics.mean(v[0] for v in vals),
                               statistics.stdev(v[0] for v in vals) if len(vals) > 1 else 0.0,
                               sum(v[2] for v in vals)) if vals else None))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def provenance_table(order, data, anchors):
    lines = ["## Case provenance", "",
             "Chains, seeds and target sizes behind every table above.", "",
             "| Case | Tier | Seeds | GT chains | chains scored | GT residues | designated sites |",
             "|---|---|---|---|---|---|---|"]
    for dataset, case in order:
        a = anchors.get((dataset, case)) or {}
        first = data.get((dataset, case), {}).get("AF3") or \
            next(iter(data.get((dataset, case), {}).values()), None)
        tier = "All-D" if dataset == "All-D" else "Local-D"
        lines.append(f"| {case} | {tier} | {first['n_seeds'] if first else 0} | "
                     f"{a.get('n_gt_chains', '')} | {first['n_scored_chains'] if first else ''} | "
                     f"{first['gt_res'] if first else ''} | {a.get('sites', '')} |")
    lines.append("")
    return lines


def enantiomer_table(order, data):
    lines = ["## Enantiomer verdict (global chirality)", "",
             "Number of seeds whose model is closer to the **mirrored** GT than to the GT "
             "(`mirTM > TM`).  Distance-only metrics cannot report this.", "",
             "| Case | GT | " + " | ".join(COND_LABEL[c] for c in CONDITIONS) + " |",
             "|---|---|" + "---|" * len(CONDITIONS)]
    for dataset, case in order:
        tier = "All-D" if dataset == "All-D" else "Local-D"
        cells = ["GT (by definition)"]
        for c in CONDITIONS:
            v = data.get((dataset, case), {}).get(c)
            cells.append(f"{v['mirror_models']}/{v['n_seeds']}" if v else "n/a")
        lines.append(f"| {case} ({tier}) | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports_master42_n30_ccdfix/per_case_metric_tables.md")
    ap.add_argument("--canonical-dir", type=Path, default=g.CANON)
    ap.add_argument("--target-config-root", type=Path,
                    default=ROOT / "configs/target_chirality")
    ap.add_argument("--local-d-target-manifest", type=Path,
                    default=ROOT / "data/local_d/local_d_curated_v1_1/"
                                 "14_correction_target_manifest.csv")
    ap.add_argument("--site-window", type=int, default=4)
    ap.add_argument("--lddt-radius", type=float, default=15.0)
    ap.add_argument("--dockq", default=g.DEFAULT_DOCKQ)
    ap.add_argument("--usalign", default=str(g.DEFAULT_USALIGN))
    ap.add_argument("--no-dockq", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(a.from_csv.open(newline="", encoding="utf-8")))
    if not rows:
        print(f"{a.from_csv}: no rows")
        return 1
    grouped = collections.defaultdict(dict)
    order = []
    for r in rows:
        key = (r["dataset"], r["target"])
        if key not in grouped:
            order.append(key)
        grouped[key].setdefault(r["condition"], []).append(r)
    data = {k: {c: cond_values(v) for c, v in conds.items()} for k, conds in grouped.items()}

    usalign = str(a.usalign) if Path(a.usalign).exists() else ""
    dockq = "" if a.no_dockq else (a.dockq if Path(a.dockq).exists() else "")
    workdir = Path(tempfile.mkdtemp(prefix="per_case_tables_"))
    anchors = {}
    for dataset, case in order:
        anchors[(dataset, case)] = gt_anchor(
            case, dataset, a.canonical_dir, a.target_config_root, a.local_d_target_manifest,
            a.site_window, a.lddt_radius, usalign, dockq, workdir)

    lines = ["# Per-case metrics: GT / AF3 / Corrected AT-AF3", "",
             "Generated by `scripts/per_case_metric_tables.py` from",
             f"`{a.from_csv.name}` ({len(rows)} per-seed rows, {len(order)} cases, 30 seeds each).",
             "One table per metric.", "",
             "**Columns.**  `GT` is the experimental reference scored against itself with the same",
             "functions - a *measured* anchor, not an assumption.  `Corrected AT-AF3` is the shipped",
             "correction output.  `AT-AF3 (extra)` is the D-conditioned prediction that the correction",
             "acts on; it is the scientifically informative condition.",
             "", "**Averages.**  Prediction cells are mean $\\pm$ SD over the 30 inference seeds; the tier",
             "rows are the equal-weight mean over the 7 All-D / 9 Local-D / 16 cases.",
             "", "Source data: `gt_metrics_per_seed.csv`; peers: `protocol_table.md`, "
             "`gt_metrics_report.md`, `gt_metrics_README.md`, `gt_metrics_selftest.md`.",
             "", "**Reading rules.**",
             "1. lDDT is mirror-blind (an exact enantiomer of the GT scores 1.0000): never use it alone",
             "   for a chirality question - the enantiomer table and the chirality table are separate",
             "   on purpose.",
             "2. TM-score is the official USalign value (`bin/USalign`, v20260920) over the same 1:1",
             "   residue correspondence; the numpy cross-check lives in `gt_metrics_per_seed.csv`.",
             "3. `coverage` is relative to the chains that are scored, not to the whole complex.",
             "4. An empty DockQ cell means no scorable interface was found - not a score of zero.  The "
             "DockQ column of this file is **not used for scoring**: DockQ does not see the `HETATM` "
             "D-residues of the references, so it scores the L-subset of each interface; the "
             "per-interface values are in `dockq_interface_report.md`.",
             "5. The chirality columns are computed from the canonical PDBs; the per-case chirality",
             "   tables read the raw run CIFs.",
             "", "**Reference-chain note.** 7P5R_2, 8AVR_2, 8BFD_2 and 8P6Q_2 deposit",
             "**both enantiomers** of the target peptide.  The target configs designate the D copy",
             "(7P5R_2 -> C, 8AVR_2 -> B, 8BFD_2 -> B, 8P6Q_2 -> B); the check table below reports",
             "the per-case verdict.  For **7W40_6**, the designated DPN is located by CCD code on",
             "the canonical chain, because the curation manifest uses source-CIF chain letters.", ""]
    flags = {(ds, case) for (ds, case), a in anchors.items() if a.get("gt_wrong", 0) > 0}
    lines += gt_check_table(order, anchors, flags)
    for title, key, note in METRICS:
        lines += case_table(title, key, note, order, data, anchors, flags)
    lines += enantiomer_table(order, data)
    lines += provenance_table(order, data, anchors)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"wrote {a.out}")
    print(f"  {len(order)} cases, {len(METRICS)} metric tables + enantiomer + provenance, "
          f"{len(lines)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


