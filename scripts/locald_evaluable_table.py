#!/usr/bin/env python3
"""Local-D target D-chirality table from the corrected-code benchmark.

Target positions come from the curated task metadata
(data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv),
NOT from the model, and not inferred from final chirality.

Non-evaluable target sites are removed from the denominator for every condition.
Chirality test: the N-CA, C-CA, CB-CA signed volume defined in docs/METRICS.md.
"""
from __future__ import annotations
import argparse, collections, csv, statistics
from pathlib import Path
import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROOTDIR = ROOT / "runs/local_d_curated_v1_1_post_cutoff"
GTDIR = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_pdb"


def ccd_chain(case: str, ccd: str, canonical_dir: Path = GTDIR):
    """Canonical-GT chains that carry the `ccd` residue (the manifest letters are source-based)."""
    path = canonical_dir / f"Local_D_{case}_seed_0_GT.pdb"
    if not path.exists():
        return []
    st = gemmi.read_structure(str(path))
    return sorted({ch.name for ch in st[0] for r in ch
                   if r.name.strip().upper() == ccd})

CASES = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1", "8U78_1", "8UM9_1", "9BIC_1"]
NEED = ("N", "CA", "C", "CB")


def find_model(cond: str, case: str, seed: str) -> Path | None:
    if cond == "AF3":
        p = ROOTDIR / "standard_af3" / case / f"seed_{seed}" / f"{case}_standard_af3" / f"{case}_standard_af3_model.cif"
    elif cond == "AT-AF3":
        p = ROOTDIR / "at_af3_first_pass" / case / f"seed_{seed}" / f"{case}_at_af3" / f"{case}_at_af3_model.cif"
    else:
        p = ROOTDIR / "corrected" / case / f"seed_{seed}" / "corrected_all_errors.cif"
    return p if p.exists() else None


def score(path: Path, targets: dict) -> tuple[int, int]:
    st = gemmi.read_structure(str(path))
    if not st:
        return 0, 0
    n_eval = n_ok = 0
    for ch in st[0]:
        want = targets.get(ch.name)
        if not want:
            continue
        for res in ch:
            if res.seqid.num not in want:
                continue
            amap = {}
            for a in res:
                amap.setdefault(a.name.strip(), a)
            if not all(k in amap for k in NEED):
                continue
            p = {k: np.array([amap[k].pos.x, amap[k].pos.y, amap[k].pos.z]) for k in NEED}
            t = float(np.dot(np.cross(p["N"] - p["CA"], p["C"] - p["CA"]), p["CB"] - p["CA"]))
            if t == 0.0:
                continue
            n_eval += 1
            if t < 0:
                n_ok += 1
    return n_eval, n_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-manifest",
                    default="data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv")
    ap.add_argument("--canonical-dir", type=Path, default=GTDIR,
                    help="directory of canonical GT PDBs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    targets = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in csv.DictReader((ROOT / a.target_manifest).open(newline="", encoding="utf-8")):
        if r["case_id"] not in CASES:
            continue
        if str(r.get("target_chirality", "D")).upper() != "D":
            continue
        # The manifest's chain letters come from the *source* CIF; the run outputs (and the
        # canonical PDBs) may use different letters.  Resolve the chain by CCD code, keep the
        # manifest's residue number.
        chain = r["chain_id"]
        hits = ccd_chain(r["case_id"], r["ccd_id"].strip().upper(), a.canonical_dir)
        if len(hits) == 1 and hits[0] != chain:
            print(f"  note: {r['case_id']} {r['ccd_id']} is on canonical chain {hits[0]} "
                  f"(manifest says {chain}) - using the CCD chain")
            chain = hits[0]
        raw = r.get("label_seq_id") or r.get("residue_id")
        try:
            targets[r["case_id"]][chain].add(int(raw))
        except (TypeError, ValueError):
            pass
    print("target sites per case:", {c: sum(len(v) for v in targets[c].values()) for c in CASES})

    COND = ["AF3", "AT-AF3", "Corrected ATAF3"]
    per = collections.defaultdict(list)
    for case in CASES:
        seeds = sorted(p.name.split("_")[1] for p in (ROOTDIR / "standard_af3" / case).glob("seed_*"))
        for seed in seeds:
            for cond in COND:
                p = find_model(cond, case, seed)
                if p is None:
                    continue
                per[(case, cond)].append(score(p, targets[case]))

    lines = ["# Local-D target D-chirality accuracy", "",
             "Evaluable-only denominator, identical for all conditions. Mean $\\pm$ sample SD over seeds.", "",
             "| Protein | AF3 | AT-AF3 | Corrected ATAF3 | evaluable n | seeds |",
             "|---|---|---|---|---|---|"]
    latex = ["\\begin{tabular}{lccc}", "\\toprule",
             "\\textbf{Protein} & \\textbf{AF3} & \\textbf{AT-AF3} & \\textbf{Corrected ATAF3} \\\\", "\\midrule"]
    for case in CASES:
        cells, n_show, nseeds = [], None, 0
        for cond in COND:
            vals = per.get((case, cond), [])
            acc = [k / n for n, k in vals if n > 0]
            ns = sorted({n for n, _ in vals if n > 0})
            nseeds = max(nseeds, len(vals))
            if n_show is None and ns:
                n_show = ns[0] if len(ns) == 1 else ns
            if acc:
                m = statistics.mean(acc)
                sd = statistics.stdev(acc) if len(acc) > 1 else 0.0
                cells.append(f"{m:.4f} $\\pm$ {sd:.4f}")
                latex.append(f"{case.replace('_', chr(92) + '_')} & " if cond == "AF3" else "")
                latex[-1] += f"${m:.4f}" + (f" \\pm {sd:.4f}$" if cond != "Corrected ATAF3" else "$")
            else:
                cells.append("n/a")
        lines.append(f"| {case} | " + " | ".join(cells) + f" | {n_show} | {nseeds} |")
        latex.append(" \\\\")
    lines += ["", "## LaTeX", ""] + latex + ["\\bottomrule", "\\end{tabular}", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
