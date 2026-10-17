#!/usr/bin/env python3
"""Chain-wise C-alpha RMSD of AF3 and Corrected-AT-AF3 against the experimental GT.

Tertiary-structure question: does Corrected-AT-AF3 change the global chain fold
relative to AF3?

Protocol (per target, per seed):
  * chains are matched through the canonical chain IDs already present in the
    canonical PDBs;
  * residues are paired *within each chain* by sequence alignment (never by raw
    residue numbering);
  * only matched C-alpha atoms are used;
  * each chain is superposed independently with the Kabsch algorithm, allowing
    only proper rotations + translations (no reflection, no mirroring).

Reported per prediction: mean chain RMSD, max chain RMSD, matched CA count.
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import secondary_structure_q3 as ss3  # noqa: E402
from gt_metrics_suite import flatten, pair_chains, target_chain_list  # noqa: E402

CANON = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_pdb"
TARGET_CFG = ROOT / "configs/target_chirality"   # all-D target-chain definitions

ALL_D = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
LOCAL_D = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1",
           "8U78_1", "8UM9_1", "9BIC_1"]
MIN_CA = 3  # fewer than 3 points cannot define a meaningful superposition
KEYS = ("RMSD_AF3", "RMSD_AT_AF3", "RMSD_Corrected_ATAF3", "delta_RMSD",
        "max_chain_RMSD_AF3", "max_chain_RMSD_Corrected_ATAF3",
        "RMSD_to_mirrored_GT_AF3", "RMSD_to_mirrored_GT_Corrected",
        "Rg_GT", "Rg_AF3", "Rg_Corrected_ATAF3", "Rg_ratio_Corrected_over_GT")


def kabsch_rmsd(mobile, target):
    """RMSD after optimal rigid-body superposition (proper rotation only)."""
    P = np.asarray(mobile, dtype=float)
    Q = np.asarray(target, dtype=float)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = (R @ Pc.T).T - Qc
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def load(path: Path):
    """(seq_by_chain, ca_by_key) using the shared residue reader."""
    order, res = ss3.protein_residues(path)
    seq, ca = {}, {}
    for k in order:
        seq.setdefault(k[0], []).append((k, ss3.one_letter(res[k]["resname"])))
        atom = res[k]["atoms"].get("CA")
        if atom is not None:
            line = atom[1]
            ca[k] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    return seq, ca


def pair_arrays(gt_path: Path, pred_path: Path, target_chains=None):
    """[(chain, model_CA, gt_CA)] for every matched chain with >= MIN_CA CA atoms.

    Chains are paired inside matching chain names (the benchmark convention), except when
    `target_chains` is given: those chains - the target chains that a
    `configs/target_chirality` config designates for an all-D case - are paired **by sequence**
    to the best model chain, so the model is scored against the **D** chain of the entry rather
    than the L enantiomer that the same entry also deposits.  See `docs/METRICS.md`.

    `max_fill=6` (not the library default 3) is required because the 8BFD_2 GT carries four
    consecutive AIB residues.
    """
    gt_seq, gt_ca = load(gt_path)
    p_seq, p_ca = load(pred_path)
    by_chain = collections.defaultdict(list)
    gt_like = {c: {"seq": gt_seq[c]} for c in gt_seq}
    pred_like = {c: {"seq": p_seq[c]} for c in p_seq}
    for gk, pk in flatten(pair_chains(gt_like, pred_like, target_chains)):
        by_chain[gk[0]].append((gk, pk))
    out = []
    for chain in sorted(by_chain):
        mob, tgt = [], []
        for gk, pk in by_chain[chain]:
            if gk in gt_ca and pk in p_ca:
                mob.append(p_ca[pk])
                tgt.append(gt_ca[gk])
        if len(mob) >= MIN_CA:
            out.append((chain, np.asarray(mob), np.asarray(tgt)))
    return out


def chainwise_rmsd(gt_path: Path, pred_path: Path, target_chains=None):
    """[(chain, n_ca, rmsd)] for every matched chain (proper rotation only)."""
    return [(ch, len(mob), kabsch_rmsd(mob, tgt))
            for ch, mob, tgt in pair_arrays(gt_path, pred_path, target_chains)]


def mirror(a: np.ndarray) -> np.ndarray:
    """Reflection through the x = 0 plane (used only as a separate reference)."""
    b = a.copy()
    b[:, 0] = -b[:, 0]
    return b


def chainwise_mirror_rmsd(gt_path: Path, pred_path: Path, target_chains=None):
    """Mean chain RMSD of the prediction onto the *mirrored* GT (proper rotation)."""
    vals = [kabsch_rmsd(mob, mirror(tgt))
            for _ch, mob, tgt in pair_arrays(gt_path, pred_path, target_chains)]
    return statistics.mean(vals) if vals else float("nan")


def radius_of_gyration(path: Path, chains=None):
    """Mean over chains of the C-alpha radius of gyration.

    `chains` restricts the average to the given chains (used to take the GT radius of gyration
    from the designated target chain instead of every copy in the entry).
    """
    seq, ca = load(path)
    vals = []
    for ch in seq:
        if chains and ch not in chains:
            continue
        X = np.asarray([ca[k] for k, _ in seq[ch] if k in ca])
        if len(X) >= MIN_CA:
            vals.append(float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())))
    return statistics.mean(vals) if vals else float("nan")


def summarise(rows):
    if not rows:
        return (float("nan"), float("nan"), 0)
    vals = [r[2] for r in rows]
    return (statistics.mean(vals), max(vals), sum(r[1] for r in rows))



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-dir", type=Path, default=CANON,
                    help="directory of canonical GT/AF3/AT-AF3/Corrected PDBs")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "reports_master42_n30_ccdfix")
    ap.add_argument("--target-config-root", type=Path, default=TARGET_CFG,
                    help="all-D target-chain definitions (see docs/METRICS.md)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for kind, cases in (("All_D", ALL_D), ("Local_D", LOCAL_D)):
        dataset = "All-D" if kind == "All_D" else "Local-D"
        for case in cases:
            files = {}
            for p in sorted(a.canonical_dir.glob(f"{kind}_{case}_seed_*_*.pdb")):
                seed, cond = p.name[:-4].split("_seed_")[1].split("_", 1)
                if cond == "GT":
                    files.setdefault("GT", p)
                else:
                    try:
                        files.setdefault(int(seed), {})[cond] = p
                    except ValueError:
                        continue
            gt = files.get("GT")
            if gt is None:
                print(f"SKIP {case}: no GT")
                continue
            # target chains: for all-D cases the config names the chain that holds the D copy
            gt_seq, _gt_ca = load(gt)
            tchains = target_chain_list(case, dataset,
                                        {c: {"seq": gt_seq[c]} for c in gt_seq},
                                        a.target_config_root)
            rg_gt = radius_of_gyration(gt, tchains)
            seeds = sorted(s for s in files if isinstance(s, int))
            print(f"[{dataset}] {case}: {len(seeds)} seeds"
                  f"{' (target chain ' + ','.join(tchains) + ')' if tchains else ''}", flush=True)
            for seed in seeds:
                rec = files[seed]
                af3, cor = rec.get("AF3"), rec.get("Corrected_AT_AF3")
                at = rec.get("AT_AF3")
                if af3 is None or cor is None:
                    continue
                a_rows = chainwise_rmsd(gt, af3, tchains)
                c_rows = chainwise_rmsd(gt, cor, tchains)
                t_rows = chainwise_rmsd(gt, at, tchains) if at is not None else []
                m_a, x_a, n_a = summarise(a_rows)
                m_c, x_c, n_c = summarise(c_rows)
                m_t, _, n_t = summarise(t_rows)
                mi_a = chainwise_mirror_rmsd(gt, af3, tchains)
                mi_c = chainwise_mirror_rmsd(gt, cor, tchains)
                rg_a = radius_of_gyration(af3)
                rg_c = radius_of_gyration(cor)
                per_seed.append({
                    "target": case, "task": dataset, "reference": "GT",
                    "seed": int(seed),
                    "matched_CA": n_a,
                    "matched_CA_Corrected": n_c,
                    "n_chains": len(a_rows),
                    "RMSD_AF3": round(m_a, 4),
                    "RMSD_AT_AF3": round(m_t, 4),
                    "RMSD_Corrected_ATAF3": round(m_c, 4),
                    "delta_RMSD": round(m_c - m_a, 4),
                    "delta_Corrected_minus_AT": round(m_c - m_t, 4),
                    "RMSD_to_mirrored_GT_AF3": round(mi_a, 4),
                    "RMSD_to_mirrored_GT_Corrected": round(mi_c, 4),
                    "closer_to_AF3": "mirror" if mi_a < m_a else "GT",
                    "closer_to_Corrected": "mirror" if mi_c < m_c else "GT",
                    "Rg_GT": round(rg_gt, 4),
                    "Rg_AF3": round(rg_a, 4),
                    "Rg_Corrected_ATAF3": round(rg_c, 4),
                    "Rg_ratio_Corrected_over_GT": round(rg_c / rg_gt, 3) if rg_gt else "",
                    "max_chain_RMSD_AF3": round(x_a, 4),
                    "max_chain_RMSD_Corrected_ATAF3": round(x_c, 4),
                })

    if not per_seed:
        print("no pairs found")
        return 1

    with (a.out_dir / "tertiary_structure_per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)

    groups = collections.defaultdict(list)
    for r in per_seed:
        groups[(r["task"], r["target"])].append(r)

    def mean_of(rows, key):
        return statistics.mean(float(r[key]) for r in rows)

    summary = []
    # GT self-reference anchor: superposing the GT onto itself gives RMSD 0
    mean_rg_gt = statistics.mean(r["Rg_GT"] for r in per_seed)
    summary.append({
        "scope": "reference", "task": "-", "target": "(GT self)",
        "n_seeds": len(per_seed),
        "matched_CA_mean": round(statistics.mean(r["matched_CA"] for r in per_seed), 1),
        "n_chains_mean": round(statistics.mean(r["n_chains"] for r in per_seed), 2),
        "RMSD_AF3": 0.0, "RMSD_AT_AF3": 0.0, "RMSD_Corrected_ATAF3": 0.0,
        "delta_RMSD": 0.0, "delta_Corrected_minus_AT": 0.0,
        "max_chain_RMSD_AF3": 0.0, "max_chain_RMSD_Corrected_ATAF3": 0.0,
        "RMSD_to_mirrored_GT_AF3": 0.0, "RMSD_to_mirrored_GT_Corrected": 0.0,
        "Rg_GT": round(mean_rg_gt, 4), "Rg_AF3": round(mean_rg_gt, 4),
        "Rg_Corrected_ATAF3": round(mean_rg_gt, 4),
        "Rg_ratio_Corrected_over_GT": 1.0,
    })
    for (task, case), rows in sorted(groups.items()):
        entry = {
            "scope": "protein", "task": task, "target": case, "n_seeds": len(rows),
            "matched_CA_mean": round(mean_of(rows, "matched_CA"), 1),
            "n_chains_mean": round(mean_of(rows, "n_chains"), 2),
        }
        for key in KEYS:
            entry[key] = round(mean_of(rows, key), 3)
        summary.append(entry)

    for label, cases in (("All-D", ALL_D), ("Local-D", LOCAL_D),
                         ("All 16 proteins", ALL_D + LOCAL_D)):
        sel = {k: v for k, v in groups.items() if k[1] in cases}
        if not sel:
            continue
        entry = {
            "scope": "overall", "task": label, "target": f"{len(sel)} proteins",
            "n_seeds": sum(len(v) for v in sel.values()),
            "matched_CA_mean": round(statistics.mean(mean_of(v, "matched_CA")
                                                      for v in sel.values()), 1),
            "n_chains_mean": round(statistics.mean(mean_of(v, "n_chains")
                                                   for v in sel.values()), 2),
        }
        for key in KEYS:
            entry[key] = round(statistics.mean(mean_of(v, key) for v in sel.values()), 3)
        summary.append(entry)

    with (a.out_dir / "tertiary_structure_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"\nper-seed rows: {len(per_seed)}")
    hdr = (f"{'dataset':9s} {'protein':10s} {'n':>4s} {'chains':>6s} {'CA':>6s} "
           f"{'RMSD_AF3':>9s} {'RMSD_COR':>9s} {'delta':>8s} {'worse':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        if row["scope"] != "protein":
            continue
        rows = groups[(row["task"], row["target"])]
        worse = sum(1 for r in rows if r["delta_RMSD"] > 0)
        print(f"{row['task']:9s} {row['target']:10s} {row['n_seeds']:4d} "
              f"{row['n_chains_mean']:6.2f} {row['matched_CA_mean']:6.0f} "
              f"{row['RMSD_AF3']:9.3f} {row['RMSD_Corrected_ATAF3']:9.3f} "
              f"{row['delta_RMSD']:+8.3f} {worse:3d}/{row['n_seeds']}")
    print("-" * len(hdr))
    for row in summary:
        if row["scope"] == "overall":
            print(f"{row['task']:17s} {'(overall)':10s} {row['n_seeds']:4d} "
                  f"{row['n_chains_mean']:6.2f} {row['matched_CA_mean']:6.0f} "
                  f"{row['RMSD_AF3']:9.3f} {row['RMSD_Corrected_ATAF3']:9.3f} "
                  f"{row['delta_RMSD']:+8.3f}")

    print("\n=== tertiary-structure diagnostics (mean over seeds) ===")
    hdr2 = (f"{'protein':10s} {'Rg_GT':>7s} {'Rg_AF3':>7s} {'Rg_COR':>7s} {'xGT':>5s} "
            f"{'mirrorRMSD_AF3':>15s} {'mirrorRMSD_COR':>15s} {'AF3~':>6s} {'COR~':>6s}")
    print(hdr2)
    print("-" * len(hdr2))
    for row in summary:
        if row["scope"] != "protein":
            continue
        rows = groups[(row["task"], row["target"])]

        def mv(k):
            return statistics.mean(float(r[k]) for r in rows)

        ca_ = "mirror" if mv("RMSD_to_mirrored_GT_AF3") < mv("RMSD_AF3") else "GT"
        cc_ = "mirror" if mv("RMSD_to_mirrored_GT_Corrected") < mv("RMSD_Corrected_ATAF3") else "GT"
        print(f"{row['target']:10s} {mv('Rg_GT'):7.2f} {mv('Rg_AF3'):7.2f} "
              f"{mv('Rg_Corrected_ATAF3'):7.2f} "
              f"{mv('Rg_Corrected_ATAF3') / mv('Rg_GT'):5.2f} "
              f"{mv('RMSD_to_mirrored_GT_AF3'):15.3f} "
              f"{mv('RMSD_to_mirrored_GT_Corrected'):15.3f} {ca_:>6s} {cc_:>6s}")

    d = [r["delta_RMSD"] for r in per_seed]
    pp = [row["delta_RMSD"] for row in summary if row["scope"] == "protein"]
    print(f"\nPer-seed delta RMSD over {len(d)} pairs: mean {statistics.mean(d):+.4f} A, "
          f"SD {statistics.stdev(d):.4f} A")
    print(f"Per-protein delta RMSD: mean {statistics.mean(pp):+.4f} A, "
          f"range {min(pp):+.3f} to {max(pp):+.3f} A")
    print(f"Proteins with delta > 0 (further from GT): "
          f"{sum(1 for x in pp if x > 0)}/{len(pp)}")
    if len(pp) > 1 and statistics.stdev(pp) > 0:
        t = statistics.mean(pp) / (statistics.stdev(pp) / len(pp) ** 0.5)
        print(f"Protein-level one-sample t on delta RMSD: {t:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
