"""Add CCTBX/Phenix restraint-based bond-type summaries to GT validation.

The existing overall GT/Phenix outputs are left untouched.  This script uses
the same manifest, CCTBX bond proxies, seed pairing and normal-approximation
95% CI convention (mean +/- 1.96 * sample SD / sqrt(N)).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_bond_geometry import make_geometry_manager

GROUPS = (
    "CA-N|PRO", "CA-N|GLY", "CA-N|other",
    "C=O|all",
    "C-CA|GLY", "C-CA|other",
    "peptide C-N|X-PRO", "peptide C-N|non-X-PRO",
    "CA-CB|ALA", "CA-CB|ILE_THR_VAL", "CA-CB|other",
)
METRICS = ("bond_rmsd_A", "bond_rmsz", "bond_outlier_rate_pct")
CONDITIONS = ("AF3", "AT-AF3", "Corrected AT-AF3", "GT")


def parent_resname(name: str) -> str:
    # This is only residue classification; ideal geometry and sigma always
    # come from the CCTBX proxy and monomer library.
    mapping = {
        "DPR": "PRO", "DAL": "ALA", "DAR": "ARG", "DAS": "ASP",
        "DCY": "CYS", "DGL": "GLU", "DGN": "GLN", "DHI": "HIS",
        "DIL": "ILE", "DLE": "LEU", "DLY": "LYS", "DME": "MET",
        "DPN": "PHE", "DSE": "SER", "DSG": "ASN", "DTH": "THR",
        "DTR": "TRP", "DTY": "TYR", "DVA": "VAL",
    }
    return mapping.get(name.upper(), name.upper())


def atom_info(atom):
    ag = atom.parent(); rg = ag.parent(); chain = rg.parent()
    return {
        "chain": str(chain.id), "resname": str(ag.resname).strip().upper(),
        "resid": str(rg.resseq).strip(), "atom": str(atom.name).strip().upper(),
    }


def residue_key(info):
    return (info["chain"], info["resid"])


def residue_sort_key(info):
    try:
        return (info["chain"], int(info["resid"]))
    except ValueError:
        return (info["chain"], info["resid"])


def classify(a, b, infos):
    # Return exactly one requested group, or None for other CCTBX proxies.
    pair = {a["atom"], b["atom"]}
    same = residue_key(a) == residue_key(b)
    if same and pair == {"CA", "N"}:
        r = parent_resname(a["resname"])
        return f"CA-N|{'PRO' if r == 'PRO' else 'GLY' if r == 'GLY' else 'other'}"
    if same and pair == {"C", "O"}:
        return "C=O|all"
    if same and pair == {"C", "CA"}:
        return f"C-CA|{'GLY' if parent_resname(a['resname']) == 'GLY' else 'other'}"
    if same and pair == {"CA", "CB"}:
        r = parent_resname(a["resname"])
        env = "ALA" if r == "ALA" else "ILE_THR_VAL" if r in {"ILE", "THR", "VAL"} else "other"
        return f"CA-CB|{env}"
    if not same and pair == {"C", "N"}:
        c = a if a["atom"] == "C" else b
        n = a if a["atom"] == "N" else b
        if c["chain"] != n["chain"]:
            return None
        try:
            adjacent = int(n["resid"]) == int(c["resid"]) + 1
        except ValueError:
            adjacent = False
        if not adjacent:
            return None
        next_name = parent_resname(n["resname"])
        return "peptide C-N|X-PRO" if next_name == "PRO" else "peptide C-N|non-X-PRO"
    return None


def finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def stats_for(path: Path):
    hierarchy, proxies = make_geometry_manager(path)
    atoms = hierarchy.atoms(); sites = atoms.extract_xyz()
    infos = [atom_info(a) for a in atoms]
    grouped = defaultdict(list)
    for proxy in proxies:
        i, j = map(int, proxy.i_seqs)
        if i == j or i >= len(atoms) or j >= len(atoms):
            continue
        a, b = infos[i], infos[j]
        if a["atom"].startswith("H") or b["atom"].startswith("H"):
            continue
        group = classify(a, b, infos)
        if group is None:
            continue
        weight = float(proxy.weight)
        ideal = float(proxy.distance_ideal)
        if weight <= 0:
            continue
        d = math.sqrt(sum((float(sites[i][q]) - float(sites[j][q])) ** 2 for q in range(3)))
        sigma = math.sqrt(1.0 / weight)
        grouped[group].append((d - ideal, (d - ideal) / sigma))
    out = {}
    for group in GROUPS:
        vals = grouped.get(group, [])
        deltas = [x[0] for x in vals]; zs = [x[1] for x in vals]
        out[group] = {
            "n_bonds": len(vals),
            "bond_rmsd_A": math.sqrt(sum(x*x for x in deltas)/len(deltas)) if deltas else math.nan,
            "bond_rmsz": math.sqrt(sum(x*x for x in zs)/len(zs)) if zs else math.nan,
            "bond_outlier_rate_pct": 100.0 * sum(abs(x) > 5 for x in zs) / len(zs) if zs else math.nan,
        }
    return out


def ci(values):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return len(values), math.nan, math.nan, math.nan
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values))
    return len(values), mean, mean - half, mean + half


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT/"results/gt_comparison_phenix/gt_phenix_manifest.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT/"results/gt_comparison_phenix/bond_type")
    args = ap.parse_args()
    manifest = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    raw = []
    cache = {}
    for n, row in enumerate(manifest, 1):
        path = Path(row["input_cif"])
        if not path.is_file():
            raise SystemExit(f"STOP missing input: {path}")
        key = str(path.resolve())
        if key not in cache:
            print(f"bond types {n}/{len(manifest)}", flush=True)
            cache[key] = stats_for(path)
        for group, values in cache[key].items():
            raw.append({"dataset_type": row["dataset_type"], "case": row["case"], "seed": int(row["seed"]), "condition": row["condition"], "bond_type": group.split("|",1)[0], "environment": group.split("|",1)[1], **values})
    write_csv(args.output_dir/"bond_type_per_structure.csv", raw)
    by = {(r["dataset_type"], r["case"], int(r["seed"]), r["bond_type"], r["environment"], r["condition"]): r for r in raw}
    keys = sorted({k[:5] for k in by if k[5] != "GT"})
    paired = []
    for dataset, case, seed, bond_type, env in keys:
        gt = by.get((dataset, case, 0, bond_type, env, "GT"))
        if not gt:
            continue
        row = {"dataset_type":dataset,"case":case,"seed":seed,"bond_type":bond_type,"environment":env}
        for cond in CONDITIONS[:3]:
            q = by[(dataset,case,seed,bond_type,env,cond)]
            for metric in METRICS:
                row[f"{cond}_{metric}"] = q[metric]
                row[f"{cond}_{metric}_abs_error_vs_gt"] = abs(float(q[metric])-float(gt[metric])) if finite(q[metric]) and finite(gt[metric]) else math.nan
        for metric in METRICS:
            row[f"AF3_minus_AT-AF3_{metric}_error"] = row[f"AF3_{metric}_abs_error_vs_gt"] - row[f"AT-AF3_{metric}_abs_error_vs_gt"] if finite(row[f"AF3_{metric}_abs_error_vs_gt"]) and finite(row[f"AT-AF3_{metric}_abs_error_vs_gt"]) else math.nan
            row[f"AT-AF3_minus_Corrected_{metric}_error"] = row[f"AT-AF3_{metric}_abs_error_vs_gt"] - row[f"Corrected AT-AF3_{metric}_abs_error_vs_gt"] if finite(row[f"AT-AF3_{metric}_abs_error_vs_gt"]) and finite(row[f"Corrected AT-AF3_{metric}_abs_error_vs_gt"]) else math.nan
        paired.append(row)
    write_csv(args.output_dir/"bond_type_paired_vs_gt.csv", paired)
    groups = defaultdict(list)
    for r in paired: groups[(r["dataset_type"],r["case"],r["bond_type"],r["environment"])].append(r)
    summary=[]
    for (dataset,case,bond_type,env), rows in sorted(groups.items()):
        z={"dataset_type":dataset,"case":case,"bond_type":bond_type,"environment":env,"n_samples":len(rows)}
        for metric in METRICS:
            for cond in CONDITIONS[:3]:
                n,m,lo,hi=ci([r[f"{cond}_{metric}_abs_error_vs_gt"] for r in rows])
                z[f"{cond}_{metric}_abs_error_vs_gt_mean"]=m; z[f"{cond}_{metric}_abs_error_vs_gt_ci95_low"]=lo; z[f"{cond}_{metric}_abs_error_vs_gt_ci95_high"]=hi; z[f"{cond}_{metric}_abs_error_vs_gt_n_valid"]=n
            for label in ("AF3_minus_AT-AF3","AT-AF3_minus_Corrected"):
                n,m,lo,hi=ci([r[f"{label}_{metric}_error"] for r in rows])
                z[f"{label}_{metric}_error_mean"]=m; z[f"{label}_{metric}_error_ci95_low"]=lo; z[f"{label}_{metric}_error_ci95_high"]=hi; z[f"{label}_{metric}_error_n_valid"]=n
        summary.append(z)
    write_csv(args.output_dir/"bond_type_per_case_summary.csv", summary)
    (args.output_dir/"summary.json").write_text(json.dumps({"comparison":"same paired GT model as overall validation","metrics":METRICS,"groups":GROUPS,"restraint_source":"Phenix/CCTBX geometry restraints and monomer library","ci":"mean +/- 1.96 * sample SD / sqrt(N)","n_structure_rows":len(raw),"n_paired_rows":len(paired),"n_summary_rows":len(summary)},indent=2)+"\n")
    print(f"complete structure_rows={len(raw)} paired_rows={len(paired)} summary_rows={len(summary)}")


if __name__ == "__main__":
    raise SystemExit(main())
