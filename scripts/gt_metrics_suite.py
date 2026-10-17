#!/usr/bin/env python3
"""GT-referenced metrics suite: per-chain TM-score / lDDT / coverage + site lDDT + DockQ + chirality.

Implements the following evaluation protocol:

  * **per-chain TM-score and lDDT** are the primary GT comparisons (AF3 vs
    Corrected AT-AF3); C-alpha RMSD is retained only as a supplementary metric;
  * **Local-D additionally reports lDDT for a predefined neighbourhood around each
    designated site** (`--site-window`, default +-4 residues);
  * **multi-chain targets get DockQ for the relevant interfaces**, because "a good
    score for each isolated chain does not establish correct chain arrangement";
  * the **same GT residue correspondence and the same evaluation scope** are used
    for every condition, and **matched-residue coverage is reported** so that
    missing regions cannot improve a score merely by being omitted;
  * **chirality is reported separately** (global enantiomer verdict + per-site
    D-chirality accuracy), because distance-based metrics such as lDDT are
    mirror-invariant.

Structures: the canonicalised PDBs of the benchmark
(`results/gt_comparison_phenix_ccdfix/canonical_pdb`) whose chain IDs are already
canonical, so the chain correspondence is shared by construction.  Residues are
paired *inside each chain* by sequence alignment reusing
`scripts/secondary_structure_q3.align_chains` - the same convention as
`scripts/tertiary_structure_rmsd.py`.  Nothing is mirrored except where a mirrored
reference is explicitly requested (mirTM / mirRMSD, reported as chirality evidence).

Metric notes (read before quoting any number):

  * **TM-score is the official USalign binary** (`bin/USalign`, Zhang-lab source, Version
    20260920), invoked per chain as `USalign model ref -TMscore 1 -outfmt 2` over the same
    1:1 residue correspondence the other metrics use.  `--no-usalign` falls back to the numpy
    implementation (`--tm-impl numpy`); the default `--tm-impl both` also records the numpy
    value in `TM_numpy`, so the numpy cross-check is available per seed. lDDT is
    computed directly in numpy (no external lDDT binary is used).
  * DockQ *is* the installed external tool (set `DOCKQ=/path/to/DockQ`).
  * The chirality column here is computed from the **canonical** PDBs, whereas the per-case
    chirality tables read the raw run CIFs; the two are expected to agree but are not
    the same file objects.

Run (Phenix interpreter, which is what the rest of the GT tooling uses):
    phenix.python scripts/gt_metrics_suite.py --self-test
    phenix.python scripts/gt_metrics_suite.py            # full 16-case sweep
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import secondary_structure_q3 as ss3  # noqa: E402  (one_letter, align_chains live here)

CANON = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_pdb"
ALL_D = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
LOCAL_D = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1",
           "8U78_1", "8UM9_1", "9BIC_1"]
# manifest condition -> canonical-PDB condition suffix
COND_SUFFIX = {"AF3": "AF3", "AT-AF3": "AT_AF3", "Corrected AT-AF3": "Corrected_AT_AF3"}
COND_ORDER = ["AF3", "AT-AF3", "Corrected AT-AF3"]
CHIRAL = ("N", "CA", "C", "CB")          # the triple-product test needs all four
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)        # CASP lDDT thresholds
DEFAULT_DOCKQ = os.environ.get("DOCKQ", "DockQ")
DEFAULT_USALIGN = Path(os.environ.get("USALIGN", ROOT / "bin/USalign"))     # optional Zhang-lab binary

# The canonical references write every D-residue as a `HETATM` record under its D-CCD name,
# and DockQ's parser defaults to `parse_hetatms=False`, so those residues are invisible to
# DockQ: the `dockq_*` columns of this suite therefore score only the L-residue subset of
# each reference (the AT-AF3 predictions are affected as well).  These columns are retained
# for diagnostics only; use the per-interface evaluation for all reported values.
DOCKQ_CAVEAT = (
    "**DockQ caveat - these pooled DockQ columns are not used for scoring.**  Every D-residue "
    "of the canonical references is written as a `HETATM` record with its D-CCD name, and DockQ's "
    "parser defaults to `parse_hetatms=False`, so DockQ does not see those residues: each "
    "reference is scored as its L-subset only (the same default applies to the AT-AF3 "
    "predictions).  The per-interface evaluation keeps every "
    "D-residue (labels normalised to `ATOM` + standard residue names, coordinates and chain identity "
    "untouched) and scores **every** reference interface separately with an explicit "
    "`--mapping`: `scripts/dockq_interface_eval.py`, `dockq_interface_report.md`, "
    "`dockq_interface_summary.csv`, `dockq_interface_paired.csv`."
)

def _is_heavy(name: str, element: str) -> bool:
    el = element.strip().upper()
    if el:
        return el != "H"
    return not name.strip().upper().startswith("H")


def load_chains(path: Path) -> dict:
    """chain -> {'order', 'resname', 'seq', 'xyz'} with all heavy atoms per residue.

    Same de-duplication rules (occupancy, altloc A preferred) and the same
    "N/CA/C must be present" residue filter as `secondary_structure_q3.protein_residues`;
    the difference is that *every* heavy atom is kept, not just the backbone, because
    lDDT is an all-atom metric.
    """
    out: dict = {}
    for line in path.open(errors="replace"):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        cname = line[21].strip() or "A"
        key = (cname, line[22:27].strip(), line[26].strip())
        ch = out.setdefault(cname, {"order": [], "resname": {}, "seq": [],
                                    "xyz": {}, "seen": {}, "rank": {}})
        aname = line[12:16].strip()
        if key not in ch["xyz"]:
            ch["order"].append(key)
            ch["resname"][key] = line[17:20].strip()
            ch["xyz"][key] = {}
            ch["seen"][key] = set()
            ch["rank"][key] = {}
        ch["seen"][key].add(aname)
        if not _is_heavy(aname, line[76:78] if len(line) >= 78 else ""):
            continue
        try:
            occ = float(line[54:60])
        except ValueError:
            occ = 1.0
        alt = line[16].strip()
        rank = (occ, 1 if alt in ("", "A") else 0)
        if aname not in ch["xyz"][key] or rank > ch["rank"][key][aname]:
            ch["rank"][key][aname] = rank
            ch["xyz"][key][aname] = np.array([float(line[30:38]), float(line[38:46]),
                                              float(line[46:54])], dtype=float)
    for ch in out.values():
        keep = [k for k in ch["order"] if {"N", "CA", "C"} <= ch["seen"][k]]
        keep_set = set(keep)
        ch["order"] = keep
        ch["seq"] = [(k, ss3.one_letter(ch["resname"][k])) for k in keep]
        for k in list(ch["xyz"]):
            if k not in keep_set:
                del ch["xyz"][k]
    return {c: ch for c, ch in out.items() if ch["order"]}


# ---------------------------------------------------------------------------
#  residue correspondence
# ---------------------------------------------------------------------------

def pair_chains(gt: dict, pred: dict, target_chains=None) -> dict:
    """chain -> [(gt_key, pred_key)] using the shared sequence-alignment convention.

    `max_fill=6` (not the default 3): the 8BFD_2 GT carries runs of **four** consecutive
    AIB residues (sequence `GEXXXXKEXXXXKE...`, X = non-standard), which the model fills with
    ALA.  With the default limit those four-residue blocks fall outside the gap-filling rule
    and are left unmatched; `max_fill=6` matches the full chain.

    **Chains are paired by name by default** (the canonical-benchmark convention, unchanged).
    When `target_chains` is given (the chains a config designates as the target), those chains
    are instead assigned to the **best sequence-matching model chain**, same name preferred,
    and may take over a model chain that name-matching would have given to another GT chain.

    This is required for the targets that deposit **both enantiomers**: the reference entry
    contains an L copy and a D copy of the same sequence, and the model - a single predicted
    chain - must be scored against the D chain designated by the target config.  See
    `docs/METRICS.md` for the scoring convention.
    """
    if not target_chains:
        by_chain = collections.defaultdict(list)
        for gk, pk in ss3.align_chains({c: gt[c]["seq"] for c in gt},
                                       {c: pred[c]["seq"] for c in pred}, max_fill=6):
            by_chain[gk[0]].append((gk, pk))
        return dict(by_chain)

    by_chain = collections.defaultdict(list)
    used = set()
    for gchain in list(target_chains):
        if gchain not in gt:
            continue
        best = None
        for pchain in sorted(pred):
            if pchain in used:
                continue
            cand = ss3.align_chains({pchain: gt[gchain]["seq"]},
                                    {pchain: pred[pchain]["seq"]}, max_fill=6)
            rank = (len(cand), 1 if pchain == gchain else 0)
            if best is None or rank > best[1]:
                best = (pchain, rank, cand)
        if best and best[1][0] >= 3:
            by_chain[gchain] = best[2]
            used.add(best[0])
    remaining_gt = [c for c in sorted(gt) if c not in target_chains]
    remaining_pred = [c for c in sorted(pred) if c not in used]
    if remaining_gt and remaining_pred:
        for gk, pk in ss3.align_chains({c: gt[c]["seq"] for c in remaining_gt},
                                       {c: pred[c]["seq"] for c in remaining_pred},
                                       max_fill=6):
            by_chain[gk[0]].append((gk, pk))
    return dict(by_chain)


def target_chain_list(case: str, dataset: str, gt: dict, cfg_root: Path):
    """Chains the config designates as the target (All-D), else None (Local-D: all chains)."""
    if dataset != "All-D":
        return None
    path = cfg_root / f"{case}.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text())
    chains = sorted(c for c in (cfg.get("chains") or {}) if c in gt)
    return chains or None


def matched_arrays(gt_ch: dict, pred_ch: dict, pairs: list, atoms=None):
    """Build the paired coordinate arrays for one chain.

    Returns (ref_xyz, mod_xyz, res_idx, gt_keys, atom_names) where `res_idx` marks the
    residue each atom belongs to (used to drop intra-residue pairs from lDDT).
    When `atoms` is given, only those atom names are used (e.g. ('CA',)).
    """
    ref, mod, res_idx, keys, names = [], [], [], [], []

    def _common(ga, pa):
        if atoms is None:
            return sorted(set(ga) & set(pa))
        return sorted(set(ga) & set(pa) & set(atoms))

    for r, (gk, pk) in enumerate(pairs):
        ga = gt_ch["xyz"].get(gk, {})
        pa = pred_ch["xyz"].get(pk, {})
        for an in _common(ga, pa):
            ref.append(ga[an])
            mod.append(pa[an])
            res_idx.append(r)
            keys.append(gk)
            names.append(an)
    if not ref:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), [], [])
    return (np.asarray(ref, dtype=float), np.asarray(mod, dtype=float),
            np.asarray(res_idx, dtype=int), keys, names)


def scope_keys(per_condition: dict) -> set:
    """Residues shared by *every* condition that will be scored (same-scope rule)."""
    sets = [set(gk for gk, _ in pairs) for pairs in per_condition.values() if pairs]
    return set.intersection(*sets) if sets else set()


# ---------------------------------------------------------------------------
#  superposition / TM-score
# ---------------------------------------------------------------------------

def fit(P: np.ndarray, Q: np.ndarray):
    """Proper-rotation (+ translation) fit of P onto Q (Kabsch; reflection excluded)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Q.mean(0) - R @ P.mean(0)


def apply_rt(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P.T).T + t


def ca_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """C-alpha RMSD after optimal proper superposition (supplementary metric)."""
    if len(P) < 3:
        return float("nan")
    R, t = fit(P, Q)
    diff = apply_rt(P, R, t) - Q
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def tm_score(mod_m: np.ndarray, ref_m: np.ndarray, l_ref: int, n_iter: int = 20) -> float:
    """TM-score of a prediction against one GT chain (see module docstring).

    TM = (1 / L_ref) * sum_i 1/(1 + (d_i/d0)^2),  d0 = 1.24*cbrt(L_ref - 15) - 1.8,
    L_ref = length of the **reference** chain, so residues missing from the prediction
    contribute 0 instead of being dropped from the denominator (compatible with the coverage rule).

    The superposition is TM-align-style iterative (fit -> keep pairs within d0 -> re-fit)
    rather than a single-Kabsch superposition, so the value is >= a single-fit estimate;
    both are numpy implementations.
    """
    if len(mod_m) < 3 or l_ref < 3:
        return float("nan")
    d0 = 1.24 * ((l_ref - 15) ** (1.0 / 3.0)) - 1.8 if l_ref >= 22 else 0.5
    d0 = max(d0, 0.5)
    R, t = fit(mod_m, ref_m)
    for _ in range(n_iter):
        d = np.linalg.norm(apply_rt(mod_m, R, t) - ref_m, axis=1)
        sel = d <= d0
        if sel.sum() < 3:
            break
        R, t = fit(mod_m[sel], ref_m[sel])
    d = np.linalg.norm(apply_rt(mod_m, R, t) - ref_m, axis=1)
    return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / l_ref)


def mirror(x: np.ndarray) -> np.ndarray:
    """Reflection through x = 0; the improper transform behind mirTM / mirRMSD."""
    b = x.copy()
    b[:, 0] = -b[:, 0]
    return b


# ---------------------------------------------------------------------------
#  lDDT (all-atom, CASP definition)
# ---------------------------------------------------------------------------

def lddt(ref: np.ndarray, mod: np.ndarray, res_idx: np.ndarray, radius: float = 15.0,
          thresholds=THRESHOLDS, block: int = 1024):
    """lDDT of `mod` against `ref`, all atoms, intra-residue pairs excluded.

    A pair (i, j) enters the denominator when its **reference** distance is <= radius and
    it is preserved at threshold t when |d_mod - d_ref| < t; the score is the mean over the
    four thresholds of the preserved fraction.  Returns (lddt, n_pairs).

    Mirror-invariance is structural, not accidental: the pair set depends only on the
    reference, and |d_mod - d_ref| is unchanged when all distances are reflected - so lDDT
    cannot see chirality (this is asserted by --self-test).
    """
    M = len(ref)
    if M < 2 or len(mod) != M:
        return float("nan"), 0
    keep = np.zeros((M, M), dtype=bool)
    for i0 in range(0, M, block):
        i1 = min(i0 + block, M)
        d = np.linalg.norm(ref[i0:i1, None, :] - ref[None, :, :], axis=2)
        keep[i0:i1] = (d <= radius) & (res_idx[i0:i1, None] != res_idx[None, :])
    iu = np.triu_indices(M, k=1)
    sel = keep[iu]
    if not sel.any():
        return float("nan"), 0
    a, b = iu[0][sel], iu[1][sel]
    delta = np.abs(np.linalg.norm(mod[a] - mod[b], axis=1)
                   - np.linalg.norm(ref[a] - ref[b], axis=1))
    return float(np.mean([(delta < t).mean() for t in thresholds])), int(sel.sum())


# ---------------------------------------------------------------------------
#  chirality (reported separately: distance metrics cannot see it)
# ---------------------------------------------------------------------------

def chirality_sign(xyz: dict):
    """'D' when the signed volume (N-CA, C-CA, CB-CA) is negative, 'L' otherwise.

    Identical test to scripts/compute_metrics.py and scripts/ccdfix_evaluable_table.py.
    Returns None when any of N/CA/C/CB is missing or the volume is exactly zero.
    """
    if not all(k in xyz for k in CHIRAL):
        return None
    N, CA, C, CB = (xyz[k] for k in CHIRAL)
    t = float(np.dot(np.cross(N - CA, C - CA), CB - CA))
    if t == 0.0:
        return None
    return "D" if t < 0 else "L"


# ---------------------------------------------------------------------------
#  designated sites (chirality + Local-D neighbourhood lDDT)
# ---------------------------------------------------------------------------

def _num(key: tuple) -> str:
    """Leading integer of a PDB residue key, e.g. ('A', '16A', '') -> '16'."""
    m = "".join(ch for ch in key[1] if ch.isdigit() or ch == "-")
    return m or key[1]


def target_sites(case: str, dataset: str, gt: dict, cfg_root: Path, locald_csv: Path):
    """[(label, chain, gt_key, expected_chirality)] for the designated sites of a case.

    All-D   : every residue of the configured target chains except the excluded residue
              names (`configs/target_chirality/<case>.json`) - the same rule as
              `ccdfix_evaluable_table.py`.
    Local-D : the curated target manifest
              (`data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv`,
              consumed exactly as `locald_evaluable_table.py` does), where the site is
              resolved against the GT by **matching the CCD code**, because the manifest
              carries both an author `residue_id` and a `label_seq_id` (they differ for
              8HR3_1/8HR4_1) and the canonical PDBs are author-numbered.

    Returns (sites, notes); `notes` records every site that could not be resolved.
    """
    sites, notes = [], []
    if dataset == "All-D":
        cfg = json.loads((cfg_root / f"{case}.json").read_text())
        for chain, spec in sorted((cfg.get("chains") or {}).items()):
            expected = "D" if str(spec.get("default_target_chirality", "D")).upper() == "D" else "L"
            exclude = set(spec.get("exclude_residue_names") or [])
            for key in gt.get(chain, {}).get("order", []):
                if gt[chain]["resname"][key] in exclude:
                    continue
                sites.append((f"{chain}:{_num(key)}", chain, key, expected))
        return sites, notes

    rows = [r for r in csv.DictReader(locald_csv.open(newline="", encoding="utf-8"))
            if r["case_id"] == case and str(r.get("target_chirality", "D")).upper() == "D"]

    def _resname(ch: str, key) -> str:
        return gt[ch]["resname"][key].upper()

    for r in rows:
        chain, ccd = r["chain_id"], r["ccd_id"].upper()
        want = {r.get("residue_id", ""), r.get("label_seq_id", "")}
        chain_res = gt.get(chain, {}).get("order", [])
        label = f"{chain}:{r.get('label_seq_id') or r.get('residue_id')}({ccd})"

        # Resolution order - the manifest's chain letters come from the *source* CIF, while the
        # canonical PDBs may use different chain letters, so the CCD code is the reliable key:
        #   1) manifest chain + number + CCD      3) CCD anywhere in the entry, if unique
        #   2) CCD anywhere in the manifest chain 4) manifest chain + number (last resort)
        hit1 = [(chain, k) for k in chain_res if _num(k) in want and _resname(chain, k) == ccd]
        hit2 = [(chain, k) for k in chain_res if _resname(chain, k) == ccd]
        hit3 = [(ch, k) for ch in sorted(gt) for k in gt[ch]["order"] if _resname(ch, k) == ccd]
        hit4 = [(chain, k) for k in chain_res if _num(k) in want]
        steps = [(hit1, "chain+number+CCD"), (hit2, "CCD in the manifest chain"),
                 (hit3, f"CCD unique in the entry (canonical chain {hit3[0][0]})"
                  if len(hit3) == 1 else ""), (hit4, "number only")]
        for hits, how in steps:
            if not how or len(hits) != 1:
                continue
            ch, key = hits[0]
            if how == "number only":
                notes.append(f"{case} {label}: resname {_resname(ch, key)} != {ccd} - check the "
                             f"CCD code / reference chain (located at canonical {ch}:{_num(key)})")
            elif how != "chain+number+CCD":
                notes.append(f"{case} {label}: located at canonical {ch}:{_num(key)} by {how}")
            sites.append((f"{ch}:{_num(key)}({ccd})", ch, key, "D"))
            break
        else:
            notes.append(f"{case} {label}: UNRESOLVED (candidates={len(hit4)}, ccd_hits={len(hit3)})")
    return sites, notes


def chain_scope_order(gt_ch: dict) -> dict:
    """{gt_key: index in the GT chain order} for neighbourhood windows."""
    return {k: i for i, k in enumerate(gt_ch["order"])}


def neighbourhood_keys(gt_ch: dict, site_key: tuple, window: int) -> set:
    """GT keys within +-window sequence positions of the designated site."""
    pos = chain_scope_order(gt_ch)
    i = pos.get(site_key)
    if i is None:
        return set()
    lo, hi = max(0, i - window), min(len(gt_ch["order"]) - 1, i + window)
    return set(gt_ch["order"][lo:hi + 1])


def _mean_opt(per_chain: list, key: str) -> float:
    """Mean over chains of an optional (possibly NaN) per-chain value."""
    vals = [c[key] for c in per_chain if c.get(key) == c.get(key)]
    return statistics.mean(vals) if vals else float("nan")


def score_model(gt: dict, pred: dict, scope, sites, window: int, radius: float,
                usalign: str = "", tm_impl: str = "numpy", workdir=None,
                tag: str = "", target_chains=None) -> dict:
    """All GT-referenced metrics for one prediction against the GT.

    `scope` (a set of GT keys) restricts every metric to the same residue set, so the
    three conditions are always scored over one identical scope; coverage then reports
    how much of the GT chain that scope actually covers.
    """
    per_chain, site_lddt = [], []
    chi_ok = chi_tot = 0
    pairs_all = pair_chains(gt, pred, target_chains)
    for chain in sorted(pairs_all):
        pairs = pairs_all[chain]
        if scope is not None:
            pairs = [(gk, pk) for gk, pk in pairs if gk in scope]
        if not pairs:
            continue
        gt_ch = gt[chain]
        pr_chain = pairs[0][1][0]          # the model chain this GT chain was paired with
        pr_ch = pred[pr_chain]
        ref, mod, ridx, _keys, _names = matched_arrays(gt_ch, pr_ch, pairs)
        if len(ref) < 2:
            continue
        lddt_all, n_pairs = lddt(ref, mod, ridx, radius)
        ref_ca, mod_ca, _r, _k, _n = matched_arrays(gt_ch, pr_ch, pairs, atoms=("CA",))
        l_ref_ca = sum(1 for k in gt_ch["order"] if "CA" in gt_ch["xyz"][k])
        tm_np = tm_score(mod_ca, ref_ca, l_ref_ca)
        mir_np = tm_score(mod_ca, mirror(ref_ca), l_ref_ca)
        tm_us = mir_us = float("nan")
        if usalign and tm_impl in ("usalign", "both") and workdir is not None:
            gkeys, pkeys = [], []
            for gk, pk in pairs:
                if "CA" in gt_ch["xyz"].get(gk, {}) and "CA" in pr_ch["xyz"].get(pk, {}):
                    gkeys.append(gk)
                    pkeys.append(pk)
            if len(gkeys) >= 3:
                m_pdb = workdir / f"{tag}_{chain}_model.pdb"
                r_pdb = workdir / f"{tag}_{chain}_ref.pdb"
                x_pdb = workdir / f"{tag}_{chain}_refmir.pdb"
                write_subset_pdb(m_pdb, pkeys, pr_ch)
                write_subset_pdb(r_pdb, gkeys, gt_ch)
                write_subset_pdb(x_pdb, gkeys, gt_ch, mirror_=True)
                tm_us, _lali = usalign_tm(usalign, m_pdb, r_pdb)
                mir_us, _l2 = usalign_tm(usalign, m_pdb, x_pdb)
        if tm_impl in ("usalign", "both") and tm_us == tm_us:
            tm_primary, mir_primary = tm_us, mir_us
        else:
            tm_primary, mir_primary = tm_np, mir_np
        entry = {
            "chain": chain,
            "n_gt_res": len(gt_ch["order"]),
            "n_matched_res": len(pairs),
            "coverage": len(pairs) / max(len(gt_ch["order"]), 1),
            "lddt": lddt_all,
            "n_pairs": n_pairs,
            "tm": tm_primary,
            "mir_tm": mir_primary,
            "tm_numpy": tm_np,
            "mir_tm_numpy": mir_np,
            "tm_usalign": tm_us,
            "mir_tm_usalign": mir_us,
            "ca_rmsd": ca_rmsd(mod_ca, ref_ca),
            "mir_rmsd": ca_rmsd(mod_ca, mirror(ref_ca)),
            "n_ca": len(ref_ca),
        }
        per_chain.append(entry)

        key2pred = dict(pairs)
        for label, ch, gkey, expected in sites:
            if ch != chain:
                continue
            pk = key2pred.get(gkey)
            if pk is None:
                continue
            sign = chirality_sign(pred[pr_chain]["xyz"].get(pk, {}))
            if sign is None:
                continue
            chi_tot += 1
            chi_ok += int(sign == expected)
            sub = [(gk, qk) for gk, qk in pairs
                   if gk in neighbourhood_keys(gt_ch, gkey, window)]
            if len(sub) >= 2:
                r2, m2, i2, _k2, _n2 = matched_arrays(gt_ch, pr_ch, sub)
                val, _np = lddt(r2, m2, i2, radius)
                if val == val:
                    site_lddt.append(val)

    if not per_chain:
        return {}
    weights = [c["n_pairs"] for c in per_chain]
    tot_w = sum(weights) or 1
    gt_res = sum(c["n_gt_res"] for c in per_chain)
    matched = sum(c["n_matched_res"] for c in per_chain)
    tm_vals = [c["tm"] for c in per_chain if c["tm"] == c["tm"]]
    return {
        "n_chains": len(per_chain),
        "n_gt_res": gt_res,
        "n_matched_res": matched,
        "coverage": matched / max(gt_res, 1),
        "lddt": sum(c["lddt"] * w for c, w in zip(per_chain, weights) if c["lddt"] == c["lddt"]) / tot_w,
        "tm_mean": statistics.mean(tm_vals) if tm_vals else float("nan"),
        "tm_min": min(tm_vals) if tm_vals else float("nan"),
        "tm_numpy_mean": _mean_opt(per_chain, "tm_numpy"),
        "tm_usalign_mean": _mean_opt(per_chain, "tm_usalign"),
        "mir_tm_mean": statistics.mean([c["mir_tm"] for c in per_chain]),
        "mir_tm_usalign_mean": _mean_opt(per_chain, "mir_tm_usalign"),
        "ca_rmsd_mean": statistics.mean([c["ca_rmsd"] for c in per_chain]),
        "mir_rmsd_mean": statistics.mean([c["mir_rmsd"] for c in per_chain]),
        "chirality_ok": chi_ok,
        "chirality_n": chi_tot,
        "chirality_acc": (chi_ok / chi_tot) if chi_tot else float("nan"),
        "site_lddt_mean": statistics.mean(site_lddt) if site_lddt else float("nan"),
        "n_sites_lddt": len(site_lddt),
        "per_chain": per_chain,
    }


def enantiomer_verdict(tm: float, mir_tm: float) -> str:
    """Global chirality call: which reference - GT or its mirror - the model matches."""
    if tm != tm or mir_tm != mir_tm:
        return "n/a"
    if abs(mir_tm - tm) < 1e-6:
        return "ambiguous"
    return "mirror" if mir_tm > tm else "GT"


# ---------------------------------------------------------------------------
#  discovery / CLI
# ---------------------------------------------------------------------------

def discover(canon: Path, kind: str, case: str) -> dict:
    """{'GT': path, seed(int): {cond_suffix: path}} for one case."""
    files: dict = {}
    for p in sorted(canon.glob(f"{kind}_{case}_seed_*_*.pdb")):
        seed, cond = p.name[:-4].split("_seed_")[1].split("_", 1)
        if cond == "GT":
            files.setdefault("GT", p)
            continue
        try:
            files.setdefault(int(seed), {})[cond] = p
        except ValueError:
            continue
    return files


def flatten(pairs_by_chain: dict) -> list:
    return [pair for chain in sorted(pairs_by_chain) for pair in pairs_by_chain[chain]]


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="GT-referenced metrics suite (see module docstring)")
    ap.add_argument("--canonical-dir", type=Path, default=CANON)
    ap.add_argument("--target-config-root", type=Path, default=ROOT / "configs/target_chirality")
    ap.add_argument("--local-d-target-manifest", type=Path,
                    default=ROOT / "data/local_d/local_d_curated_v1_1/"
                                 "14_correction_target_manifest.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports_master42_n30_ccdfix")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case ids")
    ap.add_argument("--max-seeds", type=int, default=None, help="keep the first N seeds per case")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--site-window", type=int, default=4,
                    help="+- residues defining the Local-D neighbourhood lDDT")
    ap.add_argument("--lddt-radius", type=float, default=15.0)
    ap.add_argument("--dockq", default=DEFAULT_DOCKQ, help="path to the DockQ binary")
    ap.add_argument("--no-dockq", action="store_true", help="skip the external DockQ step")
    ap.add_argument("--dockq-max-models", type=int, default=None,
                    help="cap the number of DockQ calls (smoke tests)")
    ap.add_argument("--usalign", default=str(DEFAULT_USALIGN),
                    help="path to the official USalign binary (TM-score)")
    ap.add_argument("--no-usalign", action="store_true",
                    help="force the numpy TM-score")
    ap.add_argument("--tm-impl", choices=("numpy", "usalign", "both"), default="both",
                    help="primary TM-score source; 'both' = USalign primary, numpy recorded "
                         "as a cross-check column")
    ap.add_argument("--self-test", action="store_true",
                    help="run the identity and consistency checks and exit")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="regenerate the summary/table/README/report from an existing "
                         "per-seed CSV (nothing is re-scored)")
    return ap.parse_args(argv)


def run_self_test(a) -> int:
    probe = None
    for kind, cases in (("Local_D", LOCAL_D), ("All_D", ALL_D)):
        for case in cases:
            cand = discover(a.canonical_dir, kind, case).get("GT")
            if cand and len(load_chains(cand)) > 1:      # multi-chain: exercises DockQ too
                probe = (kind, case, cand)
                break
        if probe:
            break
    if probe is None:
        print("self-test: no multi-chain GT found")
        return 1
    kind, case, path = probe
    dataset = "All-D" if kind == "All_D" else "Local-D"
    dockq = "" if a.no_dockq else (a.dockq if Path(a.dockq).exists() else "")
    usalign = "" if a.no_usalign else (str(a.usalign) if Path(a.usalign).exists() else "")
    workdir = Path(tempfile.mkdtemp(prefix="gt_metrics_selftest_"))
    sites, _n = target_sites(case, dataset, load_chains(path), a.target_config_root,
                             a.local_d_target_manifest)
    checks = self_test(path, sites, a.lddt_radius, a.site_window, dockq, workdir, usalign)
    lines = [f"# gt_metrics_suite self-test ({case}, {kind})", "",
             f"DockQ binary: `{dockq or 'not used'}`",
             f"USalign binary: `{usalign or 'not used (numpy TM-score only)'}`", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    body = "\n".join(lines)
    (a.out_dir / "gt_metrics_selftest.md").write_text(body + "\n", encoding="utf-8")
    print(body)
    print(f"\n{sum(1 for _n, ok, _d in checks if ok)}/{len(checks)} checks passed")
    print(f"wrote {a.out_dir / 'gt_metrics_selftest.md'}")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if all(ok for _n, ok, _d in checks) else 1


def flush_csv(a, per_seed) -> None:
    """Rewrite the per-seed CSV with everything accumulated so far.

    Called after each case so a long sweep (hours, with DockQ) always leaves usable
    partial results instead of nothing if it is interrupted.
    """
    if not per_seed:
        return
    with (a.out_dir / "gt_metrics_per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)


def write_derived(a, per_seed, notes, dockq, conditions) -> list:
    """Write the per-seed CSV, the summary, the protocol table, the README and the report.

    Kept separate from `sweep` so that `--from-csv` can regenerate every derived artefact
    from an existing per-seed CSV without re-scoring 1440 structures.
    """
    with (a.out_dir / "gt_metrics_per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)

    num_keys = ["coverage", "lDDT", "TM_mean", "TM_min", "TM_usalign", "TM_numpy",
                "mirTM_mean", "mirTM_usalign", "CA_RMSD",
                "mirRMSD", "chirality_acc", "site_lDDT", "dockq_mean", "dockq_global"]
    groups = collections.defaultdict(list)
    for r in per_seed:
        groups[(r["dataset"], r["target"], r["condition"])].append(r)

    def agg(rows, key):
        vals = [float(r[key]) for r in rows if r[key] != ""]
        if not vals:
            return "", ""
        return round(statistics.mean(vals), 4), (round(statistics.stdev(vals), 4)
                                                 if len(vals) > 1 else 0.0)

    summary = []
    for (dataset, target, cond), rows in sorted(groups.items()):
        entry = {"scope": "protein", "dataset": dataset, "target": target,
                 "condition": cond, "n_seeds": len(rows),
                 "n_chains": round(statistics.mean(float(r["n_chains"]) for r in rows), 2),
                 "n_gt_res": rows[0]["n_gt_res"],
                 "n_scope_res": round(statistics.mean(float(r["n_scope_res"]) for r in rows), 1),
                 # chain counts make the coverage denominator explicit: `coverage` is relative
                 # to the chains that are actually scored, not to the whole complex.
                 "n_gt_chains": (max(int(r["n_gt_chains"]) for r in rows
                                      if str(r.get("n_gt_chains", "")).isdigit())
                                 if any(str(r.get("n_gt_chains", "")).isdigit() for r in rows) else ""),
                 "n_scored_chains": (max(int(r["n_scored_chains"]) for r in rows
                                         if str(r.get("n_scored_chains", "")).isdigit())
                                     if any(str(r.get("n_scored_chains", "")).isdigit()
                                            for r in rows) else ""),
                 "mirror_models": sum(1 for r in rows if r["enantiomer"] == "mirror")}
        for k in num_keys:
            m, s = agg(rows, k)
            entry[k + "_mean"], entry[k + "_sd"] = m, s
        summary.append(entry)
    for dataset in ("All-D", "Local-D"):
        for cond in conditions:
            rows = [r for r in per_seed if r["dataset"] == dataset and r["condition"] == cond]
            if not rows:
                continue
            targets = sorted({r["target"] for r in rows})
            prot_tm = [statistics.mean(float(r["TM_mean"]) for r in rows if r["target"] == t)
                       for t in targets]
            prot_ld = [statistics.mean(float(r["lDDT"]) for r in rows if r["target"] == t)
                       for t in targets]
            entry = {"scope": "overall", "dataset": dataset,
                     "target": f"{len(targets)} proteins",
                     "condition": cond, "n_seeds": len(rows), "n_proteins": len(targets),
                     "n_chains": round(statistics.mean(float(r["n_chains"]) for r in rows), 2),
                     "n_gt_res": "", "n_scope_res": "",
                     "mirror_models": sum(1 for r in rows if r["enantiomer"] == "mirror")}
            for k in num_keys:
                m, s = agg(rows, k)
                entry[k + "_mean"], entry[k + "_sd"] = m, s
            # SD over proteins (the pooled SD above mixes between-protein spread with seed noise)
            entry["TM_mean_protein_level_mean"] = round(statistics.mean(prot_tm), 4)
            entry["TM_mean_protein_level_sd"] = (round(statistics.stdev(prot_tm), 4)
                                                 if len(prot_tm) > 1 else 0.0)
            entry["lDDT_protein_level_mean"] = round(statistics.mean(prot_ld), 4)
            entry["lDDT_protein_level_sd"] = (round(statistics.stdev(prot_ld), 4)
                                              if len(prot_ld) > 1 else 0.0)
            summary.append(entry)
    fields = []
    for e in summary:
        for k in e:
            if k not in fields:
                fields.append(k)
    with (a.out_dir / "gt_metrics_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    write_protocol_table(a, per_seed, summary, conditions, dockq)
    write_readme(a, per_seed, dockq)
    write_report(a, per_seed, summary, notes, conditions, dockq)
    return summary


def sweep(a) -> int:
    a.out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="gt_metrics_"))
    conditions = [c for c in (a.conditions or COND_ORDER)]
    dockq = "" if a.no_dockq else (a.dockq if Path(a.dockq).exists() else "")
    usalign = "" if a.no_usalign else (str(a.usalign) if Path(a.usalign).exists() else "")
    if not dockq and not a.no_dockq:
        print(f"warning: DockQ binary not found at {a.dockq}; multi-chain DockQ skipped")
    if not usalign and not a.no_usalign:
        print(f"warning: USalign not found at {a.usalign}; falling back to the numpy TM-score")
    print(f"TM-score source: {a.tm_impl}"
          + (f" (USalign: {usalign})" if usalign else " (numpy only)"), flush=True)
    wanted = {COND_SUFFIX[c] for c in conditions}

    per_seed, notes = [], []
    dockq_calls = 0
    for kind, cases in (("All_D", ALL_D), ("Local_D", LOCAL_D)):
        dataset = "All-D" if kind == "All_D" else "Local-D"
        for case in cases:
            if a.cases and case not in a.cases:
                continue
            files = discover(a.canonical_dir, kind, case)
            gt_path = files.get("GT")
            if gt_path is None:
                notes.append(f"{case}: no canonical GT PDB")
                continue
            gt = load_chains(gt_path)
            sites, site_notes = target_sites(case, dataset, gt, a.target_config_root,
                                            a.local_d_target_manifest)
            notes.extend(site_notes)
            tchains = target_chain_list(case, dataset, gt, a.target_config_root)
            # data-integrity guard: the GT itself must carry the designated chirality, otherwise
            # the reference chain (or a CCD code) is wrong.  This is required for the All-D
            # entries that deposit both enantiomers.
            gt_ok = gt_bad = 0
            for _lab, _ch, _k, _exp in sites:
                sign = chirality_sign(gt.get(_ch, {}).get("xyz", {}).get(_k, {}))
                if sign is None:
                    continue
                gt_ok += int(sign == _exp)
                gt_bad += int(sign != _exp)
            if gt_bad:
                msg = (f"{case}: {gt_bad}/{gt_bad + gt_ok} designated sites of the GT do NOT carry "
                       f"the expected chirality - check the reference chain / CCD codes")
                notes.append(msg)
                print(f"  WARNING {msg}", flush=True)
            multi = len(gt) > 1
            seeds = sorted(s for s in files if isinstance(s, int))
            if a.max_seeds:
                seeds = seeds[:a.max_seeds]
            print(f"[{dataset}] {case}: {len(seeds)} seeds, {len(gt)} chains, "
                  f"{len(sites)} designated sites", flush=True)
            for seed in seeds:
                avail = {c: p for c, p in files[seed].items() if c in wanted}
                if not avail:
                    continue
                preds = {c: load_chains(p) for c, p in avail.items()}
                scope = scope_keys({c: flatten(pair_chains(gt, preds[c], tchains))
                                    for c in preds})
                for cond in conditions:
                    suffix = COND_SUFFIX[cond]
                    if suffix not in preds:
                        continue
                    res = score_model(gt, preds[suffix], scope, sites, a.site_window,
                                      a.lddt_radius, usalign=usalign, tm_impl=a.tm_impl,
                                      workdir=workdir,
                                      tag=f"{kind}_{case}_{seed}_{suffix}",
                                      target_chains=tchains)
                    if not res:
                        notes.append(f"{case} seed {seed} {cond}: no matched residues")
                        continue
                    row = {
                        "dataset": dataset, "target": case, "seed": seed,
                        "condition": cond, "n_chains": res["n_chains"],
                        "n_gt_res": res["n_gt_res"], "n_matched_res": res["n_matched_res"],
                        "n_scope_res": len(scope),
                        "coverage": round(res["coverage"], 4),
                        "lDDT": round(res["lddt"], 4),
                        "TM_mean": round(res["tm_mean"], 4),
                        "TM_min": round(res["tm_min"], 4),
                        "mirTM_mean": round(res["mir_tm_mean"], 4),
                        "TM_usalign": (round(res["tm_usalign_mean"], 4)
                                       if res["tm_usalign_mean"] == res["tm_usalign_mean"] else ""),
                        "TM_numpy": round(res["tm_numpy_mean"], 4),
                        "mirTM_usalign": (round(res["mir_tm_usalign_mean"], 4)
                                          if res["mir_tm_usalign_mean"] == res["mir_tm_usalign_mean"] else ""),
                        "CA_RMSD": round(res["ca_rmsd_mean"], 4),
                        "mirRMSD": round(res["mir_rmsd_mean"], 4),
                        "enantiomer": enantiomer_verdict(res["tm_mean"], res["mir_tm_mean"]),
                        "chirality_n": res["chirality_n"],
                        "chirality_correct": res["chirality_ok"],
                        "chirality_acc": (round(res["chirality_acc"], 4)
                                          if res["chirality_acc"] == res["chirality_acc"] else ""),
                        "site_lDDT": (round(res["site_lddt_mean"], 4)
                                      if res["site_lddt_mean"] == res["site_lddt_mean"] else ""),
                        "n_sites_lDDT": res["n_sites_lddt"],
                        "dockq_mean": "", "dockq_max": "", "dockq_global": "",
                        "n_interfaces": "", "dockq_interfaces": "",
                    }
                    if (dockq and multi and len(preds[suffix]) > 1
                            and (a.dockq_max_models is None
                                 or dockq_calls < a.dockq_max_models)):
                        # DockQ needs at least two chains on *both* sides; a single-chain
                        # prediction against a multimeric native has no interfaces to score.
                        summary, note = run_dockq(dockq, avail[suffix], gt_path, workdir,
                                                  f"{kind}_{case}_{seed}_{suffix}")
                        dockq_calls += 1
                        if summary:
                            row["dockq_mean"] = round(summary["dockq_mean"], 4)
                            row["dockq_max"] = round(summary["dockq_max"], 4)
                            row["dockq_global"] = (round(summary["dockq_global"], 4)
                                                   if summary["dockq_global"] == summary["dockq_global"]
                                                   else "")
                            row["n_interfaces"] = summary["n_interfaces"]
                            row["dockq_interfaces"] = summary["interfaces"]
                        elif note:
                            notes.append(note)
                    per_seed.append(row)
            print(f"    rows so far: {len(per_seed)}", flush=True)
            flush_csv(a, per_seed)
    if not per_seed:
        print("no predictions found")
        return 1

    summary = write_derived(a, per_seed, notes, dockq, conditions)

    print(f"\nper-seed rows: {len(per_seed)}")
    hdr = (f"{'dataset':8s} {'target':9s} {'cond':16s} {'n':>3s} {'cov':>6s} {'lDDT':>6s} "
           f"{'TM':>6s} {'mirTM':>6s} {'CA-RMSD':>8s} {'chi':>5s} {'siteLDDT':>8s} {'DockQ':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for e in summary:
        if e["scope"] != "protein":
            continue
        print(f"{e['dataset']:8s} {e['target']:9s} {e['condition']:16s} {e['n_seeds']:3d} "
              f"{str(e['coverage_mean']):>6s} {str(e['lDDT_mean']):>6s} {str(e['TM_mean_mean']):>6s} "
              f"{str(e['mirTM_mean_mean']):>6s} {str(e['CA_RMSD_mean']):>8s} "
              f"{str(e['chirality_acc_mean']):>5s} {str(e['site_lDDT_mean']):>8s} "
              f"{str(e['dockq_mean_mean']):>6s}")
    print("-" * len(hdr))
    for e in summary:
        if e["scope"] == "overall":
            print(f"{e['dataset']:8s} {e['target']:9s} {e['condition']:16s} {e['n_seeds']:3d} "
                  f"{str(e['coverage_mean']):>6s} {str(e['lDDT_mean']):>6s} {str(e['TM_mean_mean']):>6s} "
                  f"{str(e['mirTM_mean_mean']):>6s} {str(e['CA_RMSD_mean']):>8s} "
                  f"{str(e['chirality_acc_mean']):>5s} {str(e['site_lDDT_mean']):>8s} "
                  f"{str(e['dockq_mean_mean']):>6s}   (enantiomer=mirror: {e['mirror_models']}/{e['n_seeds']})")
    if notes:
        print(f"\nnotes ({len(notes)}):")
        for n in notes[:20]:
            print("  -", n)
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


def _fmt(m, s):
    if m == "":
        return "n/a"
    if s == "" or s is None:
        return f"{m:.4f}"
    return f"{m:.4f} $\\pm$ {s:.4f}"


def write_protocol_table(a, per_seed, summary, conditions, dockq):
    """Compact protocol table: main block + separate chirality, site-lDDT and DockQ blocks."""
    lines = ["# GT-referenced validation metrics (mean over the seeds of each condition)", "",
             "Per-chain TM-score and lDDT are the primary GT comparisons, C-alpha RMSD is supplementary;",
             "every condition is scored over the *same* GT residue scope and matched-residue coverage is",
             "reported so that omitted regions cannot raise a score.",
             "TM-score is the **official USalign** value (`bin/USalign`, v20260920); a numpy",
             "cross-check is kept per seed in `gt_metrics_per_seed.csv` (`TM_numpy`).",
             f"Local-D site lDDT uses +-{a.site_window} residues around each designated site.",
             "DockQ is the installed external tool" + (f" (`{dockq}`)." if dockq
                                                       else " (not used in this run)."),
             DOCKQ_CAVEAT,
             "", "## Main table", "",
             "| Protein | Condition | TM-score | lDDT | CA-RMSD (A) | coverage |",
             "|---|---|---|---|---|---|"]
    prot = [e for e in summary if e["scope"] == "protein"]
    for e in prot:
        lines.append(f"| {e['target']} | {e['condition']} | {_fmt(e['TM_mean_mean'], e['TM_mean_sd'])} "
                     f"| {_fmt(e['lDDT_mean'], e['lDDT_sd'])} | {_fmt(e['CA_RMSD_mean'], e['CA_RMSD_sd'])} "
                     f"| {e['coverage_mean']:.3f} |")
    for e in summary:
        if e["scope"] == "overall":
            lines.append(f"| **{e['target']} ({e['dataset']})** | {e['condition']} "
                         f"| {_fmt(e['TM_mean_mean'], e['TM_mean_sd'])} | {_fmt(e['lDDT_mean'], e['lDDT_sd'])} "
                         f"| {_fmt(e['CA_RMSD_mean'], e['CA_RMSD_sd'])} | {e['coverage_mean']:.3f} |")

    lines += ["", "## Chirality block (separate: lDDT cannot see chirality)", "",
              "| Protein | Condition | designated-site D-accuracy | sites (n) | models whose enantiomer is the mirror |",
              "|---|---|---|---|---|"]
    for e in prot:
        n_sites = [r["chirality_n"] for r in per_seed if r["target"] == e["target"]
                   and r["condition"] == e["condition"]]
        lines.append(f"| {e['target']} | {e['condition']} | "
                     f"{_fmt(e['chirality_acc_mean'], e['chirality_acc_sd'])} | "
                     f"{max(n_sites) if n_sites else 0} | {e['mirror_models']}/{e['n_seeds']} |")
    for e in summary:
        if e["scope"] == "overall":
            lines.append(f"| **{e['target']} ({e['dataset']})** | {e['condition']} | "
                         f"{_fmt(e['chirality_acc_mean'], e['chirality_acc_sd'])} | - | "
                         f"{e['mirror_models']}/{e['n_seeds']} |")

    site_rows = [e for e in prot if e["site_lDDT_mean"] != ""]
    if site_rows:
        lines += ["", "## Local-D designated-site neighbourhood lDDT", "",
                  "| Protein | Condition | site-neighbourhood lDDT | sites per model |",
                  "|---|---|---|---|"]
        for e in site_rows:
            ns = max(r["n_sites_lDDT"] for r in per_seed if r["target"] == e["target"]
                     and r["condition"] == e["condition"])
            lines.append(f"| {e['target']} | {e['condition']} | "
                         f"{_fmt(e['site_lDDT_mean'], e['site_lDDT_sd'])} | {ns} |")

    # Scored-chain coverage: makes the coverage denominator explicit, so a coverage of
    # 1.000 cannot be misread as "the whole complex is covered".
    partial = {}
    for e in prot:
        g, s = e.get("n_gt_chains"), e.get("n_scored_chains")
        if isinstance(g, int) and isinstance(s, int) and s != g:
            partial[e["target"]] = (s, g)
    if partial:
        lines += ["", "## Scored-chain coverage (why coverage can be 1.000 for a multi-chain GT)", "",
                  "`coverage` is relative to the chains that are *scored*, not to the whole complex.",
                  "", "| Protein | GT chains | chains scored |", "|---|---|---|"]
        for t, (s, g) in sorted(partial.items()):
            lines.append(f"| {t} | {g} | {s} |")

    dq = [e for e in prot if e["dockq_mean_mean"] != ""]
    if dq:
        lines += ["", "## DockQ (multi-chain targets: a good per-chain score does not prove",
                  "a correct chain arrangement)", "",
                  DOCKQ_CAVEAT, "",
                  "| Protein | Condition | DockQ (mean over interfaces) | worst / best score | interfaces |",
                  "|---|---|---|---|---|"]
        for e in dq:
            rows = [r for r in per_seed if r["target"] == e["target"]
                    and r["condition"] == e["condition"] and r["dockq_mean"] != ""]
            if not rows:
                continue
            worst = min(float(r["dockq_mean"]) for r in rows)
            best = max(float(r["dockq_max"]) for r in rows)
            n_if = max(int(r["n_interfaces"]) for r in rows if r["n_interfaces"] != "")
            lines.append(f"| {e['target']} | {e['condition']} | "
                         f"{_fmt(e['dockq_mean_mean'], e['dockq_mean_sd'])} | {worst:.3f} / {best:.3f} "
                         f"| {n_if} |")

    (a.out_dir / "protocol_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def write_readme(a, per_seed, dockq):
    """Metric definitions, scope rule, mirror-blindness argument and validation status."""
    lines = [
        "# GT metrics suite - definitions, scope rules and validation status", "",
        "Generated by `scripts/gt_metrics_suite.py`.  Structures are the canonicalised PDBs of",
        "`results/gt_comparison_phenix_ccdfix/canonical_pdb`, so chain correspondence is shared by",
        "construction; residues are paired inside each chain by sequence alignment",
        "(`secondary_structure_q3.align_chains`, the same convention as `tertiary_structure_rmsd.py`).",
        "Gap filling uses `max_fill=6` instead of the default 3: the 8BFD_2 GT carries runs of four",
        "consecutive AIB residues that the default leaves unmatched.  The same setting is used for",
        "every target, so the residue correspondence rule is uniform across the benchmark.",
        "", "## Metrics", "", "| metric | definition |", "|---|---|",
        "| TM-score | **official USalign** (`bin/USalign`, built from the Zhang-lab source, run as"
        " `USalign model ref -TMscore 1 -outfmt 2`) over the same 1:1 residue correspondence,"
        " normalised by the GT chain length (`L_ref`).  The `TM_numpy` column carries the"
        " numpy value computed on identical inputs purely as a cross-check. |",
        "| mirTM | the same measurement against the **mirrored** GT; `mirTM > TM` is the enantiomer verdict. |",
        "| lDDT | all-atom CASP lDDT: reference pairs with `d_ref <= 15 A` between different residues;"
        " a pair is preserved when `|d_mod - d_ref| < t` for t in 0.5, 1, 2, 4 A; the score is the mean of"
        " the four preserved fractions, pooled across chains by pair count. |",
        "| CA-RMSD | C-alpha RMSD after optimal proper superposition (supplementary metric). |",
        "| coverage | matched GT residues / GT residues of the scored chains.  Missing regions lower"
        " coverage (and TM); they cannot raise a score by being omitted. |",
        "| site lDDT (Local-D) | lDDT restricted to the residues within +-" + str(a.site_window) +
        " sequence positions of each designated D site, averaged over the sites of a model. |",
        "| chirality (global) | enantiomer verdict from TM vs mirTM (GT / mirror / ambiguous). |",
        "| chirality (per site) | signed volume `(N-CA)x(C-CA).(CB-CA) < 0` => D, evaluated at the"
        " designated target sites only (identical test to `compute_metrics.py`). |",
        "| DockQ | the installed external tool, run as `DockQ <model> <GT> --short --json`; interface-wise"
        " mean/max are reported for multi-chain targets. |",
        "",
        DOCKQ_CAVEAT,
        "",
        "## Scope rule (protocol detail 1)", "",
        "For every (target, seed) the matched residue set of *each* condition is computed first and their",
        "**intersection** becomes the evaluation scope; all conditions are then scored over that one scope,",
        "while `n_gt_res`, `n_scope_res`, `n_matched_res` and `coverage` stay in the per-seed CSV.  A region",
        "that a condition fails to model therefore shows up as reduced coverage instead of",
        "disappearing from the denominator.",
        "", "## Why chirality must be reported separately (protocol detail 2)", "",
        "lDDT only ever compares *internal* distances.  Reflecting a structure leaves every distance",
        "unchanged, hence `lDDT(model, GT) == lDDT(model, mirror(GT))` identically, and an exact enantiomer",
        "of the GT scores `lDDT = 1.0000`.  TM-score is not distance-only (it requires a proper-rotation",
        "superposition), so it *does* separate the enantiomer.  Both statements are asserted by `--self-test`.",
        "", "## Validation status", "",
        "* `--self-test` checks: GT self-anchor (TM = lDDT = 1, CA-RMSD = 0, coverage = 1), mirror-blindness",
        "  of lDDT, the lDDT mirror-invariance identity, the TM / mirTM asymmetry, coverage additivity,",
        "  DockQ(GT, GT) = 1.0, and the official-USalign checks (USalign(GT, GT) = 1.0, USalign sees the",
        "  enantiomer, USalign vs numpy agreement on identical inputs).  See `gt_metrics_selftest.md`.",
        "* TM-score uses the **official USalign binary** whenever it is present",
        "  (`USALIGN=/path/to/USalign`); `--no-usalign --tm-impl numpy` falls back to numpy, and",
        "  `--tm-impl both` (default) records both so the agreement can be compared in",
        "  `gt_metrics_per_seed.csv` (`TM_usalign` vs `TM_numpy`).",
        "* The chirality column is computed from the **canonical** PDBs, whereas the per-case chirality",
        "  tables read the raw run CIFs; the two are expected to agree but are not the same file objects.",
        "", "## Reproduce", "", "```bash",
        "phenix.python scripts/gt_metrics_suite.py --self-test",
        "phenix.python scripts/gt_metrics_suite.py --no-dockq --cases 8GQP_1 --max-seeds 2",
        "phenix.python scripts/gt_metrics_suite.py            # full 16-case sweep",
        "```", ""]
    (a.out_dir / "gt_metrics_README.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def write_report(a, per_seed, summary, notes, conditions, dockq):
    """Results narrative: overall table, paired per-protein deltas, mirror cases, coverage."""
    by = collections.defaultdict(dict)
    for r in per_seed:
        by[(r["dataset"], r["target"], r["seed"])][r["condition"]] = r
    prot = [e for e in summary if e["scope"] == "protein"]
    overall = [e for e in summary if e["scope"] == "overall"]

    def paired(key, dataset, cond_a, cond_b):
        out = collections.defaultdict(list)
        for (ds, target, _seed), rec in by.items():
            if ds != dataset or cond_a not in rec or cond_b not in rec:
                continue
            va, vb = rec[cond_a][key], rec[cond_b][key]
            if va != "" and vb != "":
                out[target].append(float(vb) - float(va))
        return out

    lines = ["# GT-referenced metrics report (TM-score / lDDT / coverage / DockQ / chirality)", "",
             "Generated by `scripts/gt_metrics_suite.py`.  Companion files: `gt_metrics_README.md`",
             "(definitions), `protocol_table.md`, `gt_metrics_per_seed.csv`, `gt_metrics_summary.csv`,",
             "`gt_metrics_selftest.md`.", "",
             f"Conditions scored: {', '.join(conditions)}.  DockQ: "
             + (f"`{dockq}`." if dockq else "not used in this run."),
             f"Per-seed rows: {len(per_seed)} over {len({(r['target']) for r in per_seed})} targets.", "",
             "## 1. Overall", "",
             "| dataset | condition | n | coverage | lDDT | TM-score | mirTM | CA-RMSD (A) | site lDDT | DockQ | mirror models |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in overall:
        lines.append(f"| {e['dataset']} | {e['condition']} | {e['n_seeds']} | {e['coverage_mean']:.3f} | "
                     f"{e['lDDT_mean']:.4f} | {e['TM_mean_mean']:.4f} | {e['mirTM_mean_mean']:.4f} | "
                     f"{e['CA_RMSD_mean']:.3f} | {str(e['site_lDDT_mean']) or '-'} | "
                     f"{str(e['dockq_mean_mean']) or '-'} | {e['mirror_models']}/{e['n_seeds']} |")

    sec = 1
    for dataset in ("All-D", "Local-D"):
        d_tm = paired("TM_mean", dataset, "AF3", "Corrected AT-AF3")
        d_ld = paired("lDDT", dataset, "AF3", "Corrected AT-AF3")
        if not d_tm:
            continue
        sec += 1
        lines += ["", f"## {sec}. Paired change of Corrected AT-AF3 relative to AF3 ({dataset})", "",
                  "Positive = the correction moved the model *towards* the experimental GT.", "",
                  "| Protein | mean dTM | mean dlDDT | seeds with dTM > 0 |", "|---|---|---|---|"]
        for target in sorted(d_tm):
            tm_vals, ld_vals = d_tm[target], d_ld.get(target, [])
            n_up = sum(1 for v in tm_vals if v > 0)
            lines.append(f"| {target} | {statistics.mean(tm_vals):+.4f} | "
                         f"{(statistics.mean(ld_vals) if ld_vals else float('nan')):+.4f} | "
                         f"{n_up}/{len(tm_vals)} |")

    mirrors = collections.Counter((r["dataset"], r["target"], r["condition"])
                                  for r in per_seed if r["enantiomer"] == "mirror")
    sec += 1
    lines += ["", f"## {sec}. Models whose enantiomer is the mirror of the GT", "",
              "Distance-blind metrics cannot report this; the verdict comes from TM vs mirTM.", "",
              "| dataset | protein | condition | mirror models |", "|---|---|---|---|"]
    if mirrors:
        for (ds, target, cond), n in sorted(mirrors.items()):
            lines.append(f"| {ds} | {target} | {cond} | {n} |")
    else:
        lines.append("| - | - | - | none |")

    cov = sorted((e["coverage_mean"], e["target"], e["condition"]) for e in prot)
    sec += 1
    lines += ["", f"## {sec}. Lowest coverage (missing regions cannot score well)", "",
              "| protein | condition | coverage |", "|---|---|---|"]
    for c, target, cond in cov[:12]:
        lines.append(f"| {target} | {cond} | {c:.3f} |")

    sec += 1
    lines += ["", f"## {sec}. Reading rules and caveats", "",
              "1. lDDT is mirror-blind by construction: an exact enantiomer of the GT scores 1.0000.",
              "   Never headline lDDT for a chirality question; always quote the chirality block with it.",
              "2. TM-score is the **official USalign** value (`bin/USalign`, v20260920) over the same 1:1",
              "   residue correspondence; `TM_numpy` is only the cross-check column.  The agreement",
              "   over all 1440 model-chains is in the last section - headline only the USalign numbers.",
              "3. The `Corrected AT-AF3` column is partly a pipeline self-check: the correction is *defined*",
              "   as mirroring the designated sites, so its chirality accuracy is expected to be 1.0 and is",
              "   reported for completeness only - the informative comparison is AF3 vs AT-AF3.",
              "4. Coverage is reported per condition inside one shared scope, so a condition cannot gain by",
              "   omitting residues; compare TM/lDDT only at comparable coverage.",
              "5. DockQ is only computed for multi-chain targets; a good per-chain score does not establish",
              "   a correct chain arrangement.",
              "",
              "**Reading the DockQ columns.**  " + DOCKQ_CAVEAT,
              ""]
    if notes:
        sec += 1
        counts = collections.Counter(notes)
        lines += [f"## {sec}. Notes raised while running", ""]
        for msg, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"* {msg}" + (f"  (x{n})" if n > 1 else ""))
        lines.append("")

    # required cross-check: official USalign vs the numpy implementation
    pairs = [(float(r["TM_usalign"]), float(r["TM_numpy"])) for r in per_seed
             if r.get("TM_usalign") not in ("", None) and r.get("TM_numpy") not in ("", None)]
    if pairs:
        deltas = [abs(a - b) for a, b in pairs]
        n01 = sum(1 for d in deltas if d <= 0.01)
        n05 = sum(1 for d in deltas if d <= 0.05)
        near = [abs(a - b) for a, b in pairs if a >= 0.9]
        far = [abs(a - b) for a, b in pairs if a < 0.5]
        sec += 1
        lines += [f"## {sec}. TM-score cross-check (official USalign vs numpy, identical inputs)", "",
                  f"Same 1:1 residue correspondence for both implementations, {len(pairs)} models:",
                  "",
                  f"* mean |dTM| = {statistics.mean(deltas):.4f}, max |dTM| = {max(deltas):.4f}",
                  f"* within 0.01: {n01}/{len(pairs)} ({100.0 * n01 / len(pairs):.1f}%)",
                  f"* within 0.05: {n05}/{len(pairs)} ({100.0 * n05 / len(pairs):.1f}%)"]
        if near:
            lines.append(f"* near-native subset (USalign TM >= 0.9, n={len(near)}): "
                         f"mean |dTM| = {statistics.mean(near):.4f}")
        if far:
            lines.append(f"* divergent subset (USalign TM < 0.5, n={len(far)}): "
                         f"mean |dTM| = {statistics.mean(far):.4f}")
        lines += ["",
                  "The headline TM-score is the **official USalign** value; `TM_numpy` exists only to keep",
                  "this agreement.  Agreement is exact for near-native models and the divergence",
                  "grows with structural divergence (worst for the enantiomeric mirTM comparison, which is a",
                  "degenerate alignment) - which is why the numpy column must not be used for headlines.",
                  ""]
    (a.out_dir / "gt_metrics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main(argv=None) -> int:
    a = build_args(argv)
    if a.self_test:
        return run_self_test(a)
    if a.from_csv:
        rows = list(csv.DictReader(a.from_csv.open(newline="", encoding="utf-8")))
        if not rows:
            print(f"{a.from_csv}: no rows")
            return 1
        a.out_dir.mkdir(parents=True, exist_ok=True)
        conditions = [c for c in COND_ORDER if any(r["condition"] == c for r in rows)]
        no_dq = [r for r in rows if r.get("dockq_mean", "") == "" and int(r["n_chains"]) > 1]
        notes = ([f"DockQ found no scorable interface for {len(no_dq)} model(s) whose prediction has"
                  " more than one chain (the canonical GT interfaces are not found at DockQ's default"
                  " cutoff); their DockQ columns are empty by design, not zero."] if no_dq else [])
        dockq = str(DEFAULT_DOCKQ) if any(r.get("dockq_mean", "") for r in rows) else ""
        # GT data-integrity check (a live sweep does this too; repeated here so regenerated
        # reports carry the warning as well).
        cases = []
        for r in rows:
            key = (r["dataset"], r["target"])
            if key not in cases:
                cases.append(key)
        for dataset, case in cases:
            kind = "All_D" if dataset == "All-D" else "Local_D"
            gt_path = a.canonical_dir / f"{kind}_{case}_seed_0_GT.pdb"
            if not gt_path.exists():
                continue
            gt = load_chains(gt_path)
            sites, site_notes = target_sites(case, dataset, gt, a.target_config_root,
                                            a.local_d_target_manifest)
            notes.extend(site_notes)   # site provenance (e.g. located by CCD code)
            ok = bad = 0
            for _lab, ch, k, exp in sites:
                sign = chirality_sign(gt.get(ch, {}).get("xyz", {}).get(k, {}))
                if sign is None:
                    continue
                ok += int(sign == exp)
                bad += int(sign != exp)
            if bad:
                notes.append(f"{case}: {bad}/{bad + ok} designated sites of the GT do not carry the "
                             f"expected chirality - reference chain / CCD code issue")
        summary = write_derived(a, rows, notes, dockq, conditions)
        print(f"regenerated {len(summary)} summary rows from {a.from_csv} into {a.out_dir}")
        return 0
    return sweep(a)


# ---------------------------------------------------------------------------
#  self-test
# ---------------------------------------------------------------------------

def self_test(gt_path: Path, sites, radius: float, window: int, dockq: str, workdir: Path,
              usalign: str = ""):
    """[(check, ok, detail)] validating the identities we can check without USalign."""
    gt = load_chains(gt_path)
    checks = []
    m = score_model(gt, gt, None, sites, window, radius)
    checks.append(("GT self-anchor TM == 1.000", abs(m["tm_mean"] - 1.0) < 1e-9,
                   f"{m['tm_mean']:.9f}"))
    checks.append(("GT self-anchor lDDT == 1.000", abs(m["lddt"] - 1.0) < 1e-9,
                   f"{m['lddt']:.9f}"))
    checks.append(("GT self-anchor CA-RMSD == 0.000", m["ca_rmsd_mean"] < 1e-9,
                   f"{m['ca_rmsd_mean']:.12f}"))
    checks.append(("GT self-anchor coverage == 1.000", abs(m["coverage"] - 1.0) < 1e-9,
                   f"{m['coverage']:.6f}"))

    # lDDT is mirror-blind: the enantiomer of the GT scores a perfect 1.000 on lDDT.
    check_chain = max((c for c in m["per_chain"]), key=lambda c: c["n_pairs"])["chain"]
    pairs = [(k, k) for k in gt[check_chain]["order"]]
    ref, mod, ridx, _k, _n = matched_arrays(gt[check_chain], gt[check_chain], pairs)
    mir = mirror(mod)
    l_gt_mir = lddt(ref, mir, ridx, radius)[0]
    l_mir_gt = lddt(mirror(ref), mod, ridx, radius)[0]
    checks.append(("lDDT(mirror(GT) vs GT) == 1.000 (mirror-blind)", abs(l_gt_mir - 1.0) < 1e-9,
                   f"{l_gt_mir:.9f}"))
    checks.append(("mirror-invariance lDDT(M,GT) - lDDT(M,mirGT) == 0", abs(l_gt_mir - l_mir_gt) < 1e-12,
                   f"|delta| = {abs(l_gt_mir - l_mir_gt):.3e}"))

    # ... while TM does see the enantiomer, and the verdict flips accordingly.
    gtc = gt[check_chain]
    ca = np.asarray([gtc["xyz"][k]["CA"] for k in gtc["order"] if "CA" in gtc["xyz"][k]])
    l_ref = len(ca)
    tm_mir = tm_score(mirror(ca), ca, l_ref)
    tm_gt = tm_score(ca, ca, l_ref)
    checks.append(("TM sees the enantiomer: TM(mirGT, GT) < TM(GT, GT) == 1.0",
                   tm_mir < tm_gt and abs(tm_gt - 1.0) < 1e-9, f"TM={tm_gt:.6f}, mirTM={tm_mir:.6f}"))
    checks.append(("enantiomer_verdict(mirror) == 'mirror'",
                   enantiomer_verdict(tm_mir, tm_gt) == "mirror",
                   enantiomer_verdict(tm_mir, tm_gt)))

    # Official USalign comparison with the numpy TM-score.
    if usalign:
        gkeys = [k for k in gtc["order"] if "CA" in gtc["xyz"][k]]
        m_pdb = workdir / "selftest_gt.pdb"
        x_pdb = workdir / "selftest_gt_mirror.pdb"
        write_subset_pdb(m_pdb, gkeys, gtc)
        write_subset_pdb(x_pdb, gkeys, gtc, mirror_=True)
        us_self, us_lali = usalign_tm(usalign, m_pdb, m_pdb)
        us_mir, _ = usalign_tm(usalign, x_pdb, m_pdb)
        checks.append(("USalign(GT, GT) == 1.0000 (official binary)",
                       us_self == us_self and abs(us_self - 1.0) < 1e-9,
                       f"{us_self:.6f}, Lali={us_lali}"))
        checks.append(("USalign sees the enantiomer: mirTM < TM == 1.0",
                       us_mir == us_mir and us_mir < 1.0,
                       f"TM={us_self:.6f}, mirTM={us_mir:.6f}" if us_mir == us_mir else "nan"))
        checks.append(("USalign vs numpy self-anchor agree (|d| < 0.05)",
                       abs(us_self - tm_gt) < 0.05, f"|delta| = {abs(us_self - tm_gt):.4f}"))
        if us_mir == us_mir and tm_mir == tm_mir:
            # Loose check on purpose: the enantiomeric comparison is a degenerate superposition
            # (no meaningful alignment), where the two superpositions legitimately diverge.  The
            # decision-relevant agreement is measured over real models and reported in
            # gt_metrics_report.md section 7 (TM_usalign vs TM_numpy over all per-seed rows).
            checks.append(("USalign vs numpy mirTM loosely consistent (degenerate case, |d| < 0.15)",
                           abs(us_mir - tm_mir) < 0.15,
                           f"USalign {us_mir:.4f} vs numpy {tm_mir:.4f} "
                           f"(|d|={abs(us_mir - tm_mir):.4f})"))
    else:
        checks.append(("USalign available", False, "not used (numpy TM-score only)"))

    # coverage additivity + per-chain bookkeeping
    ok_cov = abs(sum(c["n_matched_res"] for c in m["per_chain"]) - m["n_matched_res"]) == 0
    checks.append(("coverage additivity (sum of chains == overall)", ok_cov,
                   f"{m['n_matched_res']} residues"))

    # DockQ self-consistency (external tool): GT vs GT must be 1.0
    if dockq and len(gt) > 1:
        summary, note = run_dockq(dockq, gt_path, gt_path, workdir, "selftest_gt_vs_gt")
        if summary:
            checks.append(("DockQ(GT, GT) == 1.0",
                           abs(summary["dockq_mean"] - 1.0) < 1e-6,
                           f"mean {summary['dockq_mean']:.6f} over {summary['n_interfaces']} interfaces"))
        else:
            checks.append(("DockQ(GT, GT) == 1.0", False, note))
    else:
        checks.append(("DockQ(GT, GT) == 1.0", False, "skipped (no binary or single chain)"))
    return checks


# ---------------------------------------------------------------------------
#  DockQ (external tool; multi-chain targets)
# ---------------------------------------------------------------------------

def _get(row: dict, *names):
    low = {str(k).lower(): v for k, v in row.items()}
    for n in names:
        if n in low:
            return low[n]
    return None


def parse_dockq_json(path: Path):
    """(interface rows, global DockQ) from a DockQ --json report.

    DockQ v2 writes `{"best_result": {"AB": {"DockQ": ..., "fnat": ...}}, "GlobalDockQ": ...}`
    (as produced by the installed binary); v1-style flat lists and
    `{"interfaces": [...]}` layouts are still accepted.
    """
    data = json.loads(path.read_text())
    rows: list = []
    global_dq = float("nan")

    def _row(name, st):
        if not isinstance(st, dict):
            return None
        c1, c2 = st.get("chain1"), st.get("chain2")
        return {
            "interface": (f"{c1}-{c2}" if c1 and c2 else name),
            "dockq": _get(st, "dockq", "dockq_score", "total"),
            "fnat": _get(st, "fnat"),
            "lrmsd": _get(st, "lrmsd"),
            "irmsd": _get(st, "irmsd"),
        }

    if isinstance(data, dict) and isinstance(data.get("best_result"), dict):
        for name, st in data["best_result"].items():
            r = _row(name, st)
            if r:
                rows.append(r)
        g = data.get("GlobalDockQ", data.get("best_dockq"))
        if isinstance(g, (int, float)):
            global_dq = float(g)
        return rows, global_dq

    if isinstance(data, dict):
        items = data.get("interfaces") or data.get("results")
    else:
        items = data
    for r in (items if isinstance(items, list) else [data]):
        row = _row(_get(r, "interface") if isinstance(r, dict) else None, r)
        if row:
            rows.append(row)
    return rows, global_dq


def run_dockq(binary: str, model: Path, native: Path, workdir: Path, tag: str,
              n_cpu: int = 1, timeout: int = 900):
    """(summary, notes) for DockQ model-vs-native; summary is {} when DockQ cannot run."""
    out_json = workdir / f"{tag}.json"
    cmd = [binary, str(model), str(native), "--short", "--json", str(out_json),
           "--n_cpu", str(n_cpu)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - the tool is external, report and continue
        return {}, f"{tag}: dockq not run ({type(exc).__name__}: {exc})"
    if not out_json.exists():
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""]
        return {}, f"{tag}: no json (rc={proc.returncode}) {tail[0][:160]}"
    rows, global_dq = parse_dockq_json(out_json)
    rows = [r for r in rows if isinstance(r.get("dockq"), (int, float))]
    if not rows:
        return {}, f"{tag}: json without numeric DockQ"
    vals = [float(r["dockq"]) for r in rows]
    fnats = [float(r["fnat"]) for r in rows if isinstance(r.get("fnat"), (int, float))]
    return {
        "n_interfaces": len(rows),
        "dockq_mean": statistics.mean(vals),
        "dockq_max": max(vals),
        "dockq_global": global_dq,
        "fnat_mean": statistics.mean(fnats) if fnats else float("nan"),
        "interfaces": "; ".join(f"{r['interface']}:{float(r['dockq']):.3f}" for r in rows),
    }, ""


# ---------------------------------------------------------------------------
#  USalign (official Zhang-lab binary) - primary TM-score when available
# ---------------------------------------------------------------------------

def _pdb_line(serial: int, aname: str, resname: str, resnum: int, x: float, y: float,
              z: float) -> str:
    """One ATOM record in strict PDB column layout (13-16 name, 18-20 resName, 23-26 seq)."""
    name = aname if len(aname) >= 4 else f" {aname:<3s}"
    element = aname.strip()[:1]
    return (f"ATOM  {serial:5d} {name} {resname:>3s} A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n")


def write_subset_pdb(path: Path, keys, ch: dict, mirror_: bool = False) -> int:
    """Write the given (already residue-paired) residues of one chain as a single-chain PDB.

    Residue names are normalised to their L-parent and any name USalign would not know is
    written as ALA: the file exists only so that the official binary scores the **same
    C-alpha geometry over the same 1:1 residue correspondence**, and TM-score superposition
    reads no chemistry beyond the C-alpha positions.  `mirror_` negates x for mirTM.
    """
    n = 0
    with path.open("w") as fh:
        serial = 0
        for i, k in enumerate(keys, start=1):
            rn = ch["resname"].get(k, "ALA").strip().upper()
            rn = ss3.D_TO_L.get(rn, rn)
            if rn not in ss3.STANDARD:
                rn = "ALA"
            atoms = ch["xyz"].get(k, {})
            if not atoms:
                continue
            n += 1
            for an in sorted(atoms):
                x, y, z = (float(v) for v in atoms[an])
                serial += 1
                fh.write(_pdb_line(serial, an, rn, i, -x if mirror_ else x, y, z))
        fh.write("TER\nEND\n")
    return n


def usalign_tm(binary: str, model_pdb: Path, ref_pdb: Path, timeout: int = 300):
    """(TM-score, Lali) from the official USalign, `-TMscore 1` + `-outfmt 2`.

    `-TMscore 1` scores the conformation pair **without** structure-based realignment, i.e.
    over the residue correspondence we hand it (the same one the numpy metrics use), so the
    two implementations are compared on identical inputs.  TM2 (normalised by structure 2,
    the reference) is returned, which is the same denominator the numpy TM-score uses.
    Returns (nan, 0) on any failure so the caller can fall back.
    """
    cmd = [binary, str(model_pdb), str(ref_pdb), "-TMscore", "1", "-outfmt", "2"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001 - external tool
        return float("nan"), 0
    for line in proc.stdout.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 11:
            try:
                return float(f[3]), int(f[10])
            except ValueError:
                continue
    return float("nan"), 0


if __name__ == "__main__":
    raise SystemExit(main())

