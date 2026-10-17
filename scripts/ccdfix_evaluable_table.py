#!/usr/bin/env python3
"""All-D target D-chirality accuracy restricted to EVALUABLE target sites.

Non-evaluable sites (a target residue whose CA chirality cannot be resolved, e.g.
a residue built without a CB, or without a normal backbone) are removed from the
denominator for EVERY condition, so AF3, AT-AF3 and Corrected are scored over the
same site set.

Chirality test (the signed volume defined in docs/METRICS.md):
    triple = dot(cross(N-CA, C-CA), CB-CA);  D if triple < 0 else L
A site is evaluable when N, CA, C and CB are all present (GLY is never a target).

Usage:
    python scripts/ccdfix_evaluable_table.py \
        --manifest results/gt_comparison_phenix_ccdfix/gt_phenix_manifest.csv \
        --target-config-root configs/target_chirality \
        --out reports_master42_n30_ccdfix/table_all_d_chirality_evaluable.md
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

CASE_ORDER = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
COND = [("AF3", ("AF3",)),
        ("AT-AF3", ("AT-AF3",)),
        ("Corrected ATAF3", ("Corrected AT-AF3",))]
NEED = ("N", "CA", "C", "CB")
ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def target_spec(root: Path, case: str) -> list[tuple[str, str, set]]:
    """[(chain, expected_chirality, excluded_residue_names)] from the task config."""
    path = root / f"{case}.json"
    if not path.exists():
        return [("A", "D", {"GLY"})]
    cfg = json.loads(path.read_text())
    out = []
    for chain in sorted(cfg.get("chains") or {}):
        spec = cfg["chains"][chain]
        expected = "D" if str(spec.get("default_target_chirality", "D")).upper() == "D" else "L"
        out.append((chain, expected, set(spec.get("exclude_residue_names") or [])))
    return out or [("A", "D", {"GLY"})]


def score(path: Path, spec: list[tuple[str, str, set]]) -> tuple[int, int]:
    """Return (n_evaluable, n_correct) over the target sites of one structure."""
    st = gemmi.read_structure(str(path))
    if not st:
        return 0, 0
    n_eval = n_ok = 0
    for chain, expected, exclude in spec:
        for ch in st[0]:
            if ch.name != chain:
                continue
            for res in ch:
                if res.name in exclude:
                    continue
                amap = {}
                for a in res:
                    amap.setdefault(a.name.strip(), a)
                if not all(k in amap for k in NEED):
                    continue
                p = {k: np.array([amap[k].pos.x, amap[k].pos.y, amap[k].pos.z]) for k in NEED}
                triple = float(np.dot(np.cross(p["N"] - p["CA"], p["C"] - p["CA"]), p["CB"] - p["CA"]))
                if triple == 0.0:
                    continue
                n_eval += 1
                if (triple < 0 and expected == "D") or (triple > 0 and expected == "L"):
                    n_ok += 1
    return n_eval, n_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target-config-root", default="configs/target_chirality")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, newline="", encoding="utf-8")))
    present = sorted({r["condition"] for r in rows})
    print("manifest conditions:", present)

    per_seed = defaultdict(list)   # (case, label) -> [ (n_eval, n_correct) ]
    for r in rows:
        if r["dataset_type"] != "All-D":
            continue
        for label, keys in COND:
            if r["condition"] in keys:
                chain_spec = target_spec(Path(a.target_config_root), r["case"])
                ne, nk = score(resolve(r["input_cif"]), chain_spec)
                per_seed[(r["case"], label)].append((ne, nk))

    lines = []
    lines.append("# All-D target D-chirality accuracy - evaluable sites only")
    lines.append("")
    lines.append("Non-evaluable target sites are removed from the denominator for **every**")
    lines.append("condition, so all three columns share one site set. Mean $\\pm$ sample SD over")
    lines.append("30 inference seeds; each seed contributes n_correct/n_evaluable.")
    lines.append("")
    lines.append("| Protein | " + " | ".join(l for l, _ in COND) + " | evaluable n |")
    lines.append("|---|" + "---|" * (len(COND) + 1))
    stats_out = {}
    eval_n = {}
    for case in CASE_ORDER:
        cells, ns = [], []
        for label, _ in COND:
            vals = per_seed.get((case, label), [])
            acc = [k / n for n, k in vals if n > 0]
            ns.append(sorted({n for n, _ in vals}))
            if acc:
                m = statistics.mean(acc)
                sd = statistics.stdev(acc) if len(acc) > 1 else 0.0
                stats_out[(case, label)] = (m, sd, len(acc))
                cells.append(f"{m:.4f} $\\pm$ {sd:.4f}")
            else:
                cells.append("n/a")
        eq = "yes" if len({tuple(x) for x in ns}) == 1 else "**NO**"
        ndisp = ns[0][0] if ns and len(ns[0]) == 1 else ns
        eval_n[case] = ndisp
        lines.append(f"| {case} | " + " | ".join(cells) + f" | {ndisp} ({eq}) |")
    lines.append("")
    lines.append("## LaTeX")
    lines.append("")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Protein} & \\textbf{AF3} & \\textbf{AT-AF3} & \\textbf{Corrected ATAF3} \\\\")
    lines.append("\\midrule")
    for case in CASE_ORDER:
        cells = []
        for label, _ in COND:
            if (case, label) not in stats_out:
                cells.append("n/a")
                continue
            m, sd, _ = stats_out[(case, label)]
            cells.append(f"$\\mathbf{{{m:.4f}}}$" if label.startswith("Corrected")
                         else f"${m:.4f} \\pm {sd:.4f}$")
        lines.append(f"{case.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
