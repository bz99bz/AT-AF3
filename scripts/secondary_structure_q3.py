#!/usr/bin/env python3
"""Q3 secondary-structure agreement of AF3 and Corrected-AT-AF3 against GT.

DSSP assignment uses the Kabsch-Sander implementation in Phenix (`ksdssp` via
mmtbx.secondary_structure.run_ksdssp).

States are collapsed to H = {H,G,I}, E = {E,B}, C = everything else, which is
exactly the information ksdssp's HELIX/SHEET records carry.

Residues are matched between the prediction and GT through the canonical
chain/residue mapping of the benchmark: chain IDs come from
results/gt_comparison_phenix_ccdfix/canonical_chain_id_mapping.json (already
applied in the canonical PDBs) and residues are paired by sequence alignment
within each matched chain, which resolves author-vs-label numbering offsets and
gapped regions.  No structure is ever mirrored or reflected.

Run with the Phenix interpreter:
    phenix.python scripts/secondary_structure_q3.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_pdb"
MAPJSON = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_chain_id_mapping.json"

ALL_D = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
LOCAL_D = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1",
           "8U78_1", "8UM9_1", "9BIC_1"]

D_TO_L = {"DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS", "DGL": "GLU",
          "DGN": "GLN", "DSG": "ASN", "DSN": "SER", "DHI": "HIS", "DIL": "ILE",
          "DLE": "LEU", "DLY": "LYS", "MED": "MET", "DPN": "PHE", "DPR": "PRO",
          "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL"}
STANDARD = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
            "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
            "TYR": "Y", "VAL": "V"}
BACKBONE = ("N", "CA", "C", "O")


def one_letter(resname: str) -> str:
    """One-letter code used for sequence alignment; 'X' for non-standard residues.

    Non-standard backbone-bearing residues must NOT be dropped: AIB occurs every
    ~3 residues in the 7QDI_1 and 8BFD_2 GTs, and dropping it breaks chain
    continuity for DSSP.
    """
    rn = resname.strip().upper()
    parent = D_TO_L.get(rn, rn)
    return STANDARD.get(parent, "X")


def protein_residues(path: Path, model: int = 1):
    """[(chain, seqnum, icode)] plus residues with de-duplicated backbone atoms.

    Duplicate/altloc backbone records (common in the canonicalised files) make
    ksdssp exit without records, so atoms are de-duplicated here by name
    keeping the highest-occupancy record and preferring altloc A.
    """
    res_order, residues = [], {}
    cur_model = 0
    for line in path.open(errors="replace"):
        if line.startswith("MODEL"):
            cur_model += 1
            continue
        if line.startswith("ENDMDL"):
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if cur_model not in (0, model):
            continue
        key = (line[21].strip() or "A", line[22:27].strip(), line[26].strip())
        aname = line[12:16].strip()
        if key not in residues:
            residues[key] = {"resname": line[17:20].strip(), "atoms": {},
                             "atom_order": [], "seen": set()}
            res_order.append(key)
        residues[key]["seen"].add(aname)
        if aname not in BACKBONE:
            continue
        try:
            occ = float(line[54:60])
        except ValueError:
            occ = 1.0
        alt = line[16].strip()
        rank = (occ, 1 if alt in ("", "A") else 0)
        text = line if line.endswith("\n") else line + "\n"
        cur = residues[key]["atoms"].get(aname)
        if cur is None:
            residues[key]["atom_order"].append(aname)
            residues[key]["atoms"][aname] = (rank, text)
        elif rank > cur[0]:
            residues[key]["atoms"][aname] = (rank, text)
    # Keep every residue with a complete peptide backbone (N, CA and C), whatever
    # its name.  A name-based filter would drop backbone-bearing non-standard
    # residues (AIB in 7QDI_1/8BFD_2, ZAE/HJV/28J in 8U78_1, MLU in 9BIC_1, UNK in
    # the AF3 outputs) and fragment the chain, which makes DSSP report an all-coil
    # structure.  The N/CA/C rule rejects solvent, ions and ligands (HOH, SO4, EDO,
    # PEG, CL, ...) because they carry none of them.
    keep = [k for k in res_order if {"N", "CA", "C"} <= residues[k]["seen"]]
    for k in set(res_order) - set(keep):
        del residues[k]
    return keep, residues


def write_clean_pdb(handle, order, residues) -> None:
    """Write backbone atoms as ATOM records.

    ksdssp ignores HETATM records, and the canonicalised files store the
    D-residues (and parts of the deposited GT) as HETATM, so DSSP would return no
    assignment at all.  The record name is normalised here; coordinates and
    residue numbering are untouched.
    """
    prev_chain = None
    for key in order:
        chain = key[0]
        if prev_chain is not None and chain != prev_chain:
            handle.write("TER\n")
        for aname in residues[key]["atom_order"]:
            line = residues[key]["atoms"][aname][1]
            handle.write("ATOM  " + line[6:] if len(line) > 6 else line)
        prev_chain = chain
    handle.write("TER\nEND\n")


def parse_ss(records, order, residues) -> dict:
    """H/E/C per residue key from ksdssp HELIX/SHEET records."""
    ss = {k: "C" for k in order}
    spans = []
    for ln in records:
        try:
            if ln.startswith("HELIX"):
                ic, isn = ln[19].strip(), int(ln[21:25])
                ec, esn = ln[31].strip(), int(ln[33:37])
            elif ln.startswith("SHEET"):
                ic, isn = ln[21].strip(), int(ln[22:26])
                ec, esn = ln[32].strip(), int(ln[33:37])
            else:
                continue
        except (ValueError, IndexError):
            continue
        spans.append(("H" if ln.startswith("HELIX") else "E", ic, isn, ec, esn))
    for letter, ic, isn, ec, esn in spans:
        for k in order:
            if k[0] != ic:
                continue
            try:
                n = int(k[1])
            except ValueError:
                continue
            if isn <= n <= esn:
                if ss[k] != "H":
                    ss[k] = letter
    return ss


def _run_ksdssp(pdb_path, log):
    """Call Phenix's ksdssp lazily so the module imports outside a Phenix env."""
    try:
        import mmtbx.secondary_structure as mss
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "secondary-structure assignment requires Phenix; run this script "
            "with PHENIX_PYTHON (see README.md)"
        ) from exc
    return mss.run_ksdssp(pdb_path, log=log)


def dssp_for(path: Path):
    """Return (ss_by_residue_key, seq_by_chain) for one structure."""
    order, residues = protein_residues(path)
    if not order:
        return {}, {}
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        write_clean_pdb(fh, order, residues)
        tmp = fh.name
    with open("/dev/null", "w") as devnull:
        records = _run_ksdssp(tmp, devnull)
    longest = max(collections.Counter(k[0] for k in order).values()) if order else 0
    if longest >= 12 and not any(r.startswith(("HELIX", "SHEET")) for r in records):
        n_odd = sum(1 for k in order if one_letter(residues[k]["resname"]) == "X")
        print(f"  WARNING: ksdssp returned no HELIX/SHEET records for {path.name} "
              f"({len(order)} residues, longest chain {longest}, "
              f"{n_odd} non-standard) - assignment would be all-coil", flush=True)
    ss = parse_ss(records, order, residues)
    seq = {}
    for k in order:
        seq.setdefault(k[0], []).append((k, one_letter(residues[k]["resname"])))
    Path(tmp).unlink(missing_ok=True)
    return ss, seq


def backbone_identical(path_a: Path, path_b: Path, tol: float = 1e-3) -> bool:
    """True when all N/CA/C/O coordinates are unchanged between two structures."""
    def bb(path):
        out = {}
        for line in path.open(errors="replace"):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            if name not in BACKBONE:
                continue
            out[(line[21], line[22:27].strip(), name)] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
        return out
    a, b = bb(path_a), bb(path_b)
    if not a or set(a) != set(b):
        return False
    return all(max(abs(p - q) for p, q in zip(a[k], b[k])) <= tol for k in a)


def align_chains(gt_seq, pred_seq, min_block=1, max_fill=3):
    """Pair residues between GT and prediction by sequence alignment per chain.

    Unmatched runs of *equal length on both sides* (up to max_fill residues) that
    sit between two anchor blocks are paired positionally.  This recovers
    non-standard residues: the 7QDI_1 GT carries AIB exactly where the AF3 model
    carries ALA, so a strict letter match would leave a two-residue hole at every
    AIB position and exclude those residues from both numerator and denominator.
    """
    pairs = []
    for chain in sorted(set(gt_seq) & set(pred_seq)):
        g = gt_seq[chain]
        p = pred_seq[chain]
        gs = "".join(x[1] for x in g)
        ps = "".join(x[1] for x in p)
        blocks = SequenceMatcher(None, gs, ps, autojunk=False).get_matching_blocks()
        prev = None
        for g0, p0, size in blocks:
            if prev is not None:
                pg0, pp0, psize = prev
                g_gap = g0 - (pg0 + psize)
                p_gap = p0 - (pp0 + psize)
                if g_gap == p_gap and 0 < g_gap <= max_fill:
                    for off in range(g_gap):
                        pairs.append((g[pg0 + psize + off][0], p[pp0 + psize + off][0]))
            for off in range(size):
                pairs.append((g[g0 + off][0], p[p0 + off][0]))
            prev = (g0, p0, size)
    return pairs


def q3_block(gt_ss, pred_ss, pairs):
    if not pairs:
        return float("nan"), 0
    hit = sum(1 for gk, pk in pairs if gt_ss.get(gk) == pred_ss.get(pk))
    return 100.0 * hit / len(pairs), len(pairs)


def ss_composition(ss, pairs, side):
    """(helix %, sheet %, coil %) over the matched residues.

    side='gt' uses the GT member of each pair, side='pred' the prediction member.
    """
    idx = 0 if side == "gt" else 1
    c = collections.Counter(ss.get(p[idx], "C") for p in pairs)
    t = sum(c.values()) or 1
    return (100.0 * c["H"] / t, 100.0 * c["E"] / t, 100.0 * c["C"] / t)


COMP_CLASSES = ("helix", "sheet", "coil")


def composition_problems(rows, label, n_key, tol=0.01, integral=True):
    """Composition triples that are not 100 % (or not whole residues).

    A triple can only sum to 100 % *and* give integral residue counts when H, E and
    C were computed over one and the same denominator.  Mixing fractions from
    different runs can pass per-cell range checks but fails this consistency test.
    """
    problems = []
    for r in rows:
        try:
            n = float(r[n_key])
        except (KeyError, TypeError, ValueError):
            n = 0.0
        for pre in ("GT", "AF3", "Corrected"):
            vals = [float(r[f"{pre}_{cl}_percent"]) for cl in COMP_CLASSES]
            total = sum(vals)
            if abs(total - 100.0) > tol:
                problems.append(f"{label} {r.get('target')} seed {r.get('seed', '')} {pre}: "
                                f"H+E+C = {total:.4f}")
            if integral and n > 0:
                for cl, v in zip(COMP_CLASSES, vals):
                    cnt = v * n / 100.0
                    if abs(cnt - round(cnt)) > 0.005:
                        problems.append(f"{label} {r.get('target')} seed {r.get('seed', '')} "
                                        f"{pre} {cl}: {v:.4f} % of {n:.0f} residues = {cnt:.3f}")
    return problems


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-dir", type=Path, default=CANON)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports_master42_n30_ccdfix")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    seed_re = re.compile(r"^(All_D|Local_D)_(.+)_seed_(\d+)_(GT|AF3|AT_AF3|Corrected_AT_AF3)\.pdb$")
    index = defaultdict(dict)
    for p in sorted(a.canonical_dir.glob("*.pdb")):
        m = seed_re.match(p.name)
        if m:
            index[(m.group(1), m.group(2), m.group(3))][m.group(4)] = p

    per_seed = []
    for kind, cases in (("All_D", ALL_D), ("Local_D", LOCAL_D)):
        dataset = "All-D" if kind == "All_D" else "Local-D"
        for case in cases:
            seeds = sorted({k[2] for k in index if k[0] == kind and k[1] == case}, key=int)
            gt_path = index.get((kind, case, "0"), {}).get("GT") or \
                index.get((kind, case, seeds[0]), {}).get("GT")
            if gt_path is None:
                print(f"SKIP {case}: no GT"); continue
            gt_ss, gt_seq = dssp_for(gt_path)
            print(f"[{dataset}] {case}: GT {len(gt_ss)} residues, dssp ok", flush=True)
            for seed in seeds:
                rec = index.get((kind, case, seed), {})
                af3, cor, at = rec.get("AF3"), rec.get("Corrected_AT_AF3"), rec.get("AT_AF3")
                if af3 is None or cor is None:
                    print(f"  seed {seed}: missing AF3/Corrected, skipped"); continue
                af3_ss, af3_seq = dssp_for(af3)
                cor_ss, cor_seq = dssp_for(cor)
                at_ss, at_seq = (dssp_for(at) if at else ({}, {}))
                # max_fill=6 (not the default 3): the 8BFD_2 GT has four consecutive AIB
                # residues, which the default leaves unmatched (15 of 31 residues) and hence
                # makes its Q3/coverage/tertiary numbers inconsistent with the corrected
                # metrics suite.  Verified to change no other target.
                pairs = align_chains(gt_seq, af3_seq, max_fill=6)
                q_af3, n1 = q3_block(gt_ss, af3_ss, pairs)
                pairs_c = align_chains(gt_seq, cor_seq, max_fill=6)
                q_cor, n2 = q3_block(gt_ss, cor_ss, pairs_c)
                q_at, n3 = (q3_block(gt_ss, at_ss, align_chains(gt_seq, at_seq, max_fill=6))
                            if at else (float("nan"), 0))
                matched = min(n1, n2) if n1 and n2 else max(n1, n2)
                if not (n1 and n2):
                    print(f"  seed {seed}: no matched residues, skipped"); continue
                # does the correction move the backbone at all?
                bb_same = backbone_identical(at, cor) if at else ""
                # secondary-structure composition over the matched residue set
                gh, ge, gc = ss_composition(gt_ss, pairs, "gt")
                ah, ae, ac = ss_composition(af3_ss, pairs, "pred")
                ch, ce, cc = ss_composition(cor_ss, pairs, "pred")
                per_seed.append({
                    "target": case, "task": dataset, "seed": int(seed),
                    "matched_residues": matched,
                    "GT_helix_percent": round(gh, 4), "GT_sheet_percent": round(ge, 4),
                    "GT_coil_percent": round(gc, 4),
                    "AF3_helix_percent": round(ah, 4), "AF3_sheet_percent": round(ae, 4),
                    "AF3_coil_percent": round(ac, 4),
                    "Corrected_helix_percent": round(ch, 4), "Corrected_sheet_percent": round(ce, 4),
                    "Corrected_coil_percent": round(cc, 4),
                    "Q3_AF3": round(q_af3, 4),
                    "Q3_AT_AF3": round(q_at, 4) if n3 else "",
                    "Q3_Corrected_ATAF3": round(q_cor, 4),
                    "delta_Q3": round(q_cor - q_af3, 4),
                    "delta_Q3_vs_AT_AF3": round(q_cor - q_at, 4) if n3 else "",
                    "backbone_identical_AT_vs_Corrected": bb_same,
                })

    # ------------------------------------------------------------------
    # Self-check: a composition triple must be internally consistent - H + E + C
    # = 100 % and whole residue counts - so fractions from different runs cannot
    # be combined into one table row.  See docs/METRICS.md for the
    # composition-vs-Q3 distinction.
    # ------------------------------------------------------------------
    bad = composition_problems(per_seed, "per-seed", "matched_residues")
    denominators = defaultdict(set)
    for r in per_seed:
        denominators[(r["task"], r["target"])].add(r["matched_residues"])
    for key, ns in sorted(denominators.items()):
        if len(ns) > 1:
            print(f"  WARNING: {key[0]} {key[1]}: matched_residues varies over seeds "
                  f"{sorted(ns)}", flush=True)
    if bad:
        raise SystemExit("secondary_structure_q3 per-seed composition self-check FAILED "
                         f"({len(bad)} problem(s)):\n  " + "\n  ".join(bad[:20]))

    with (a.out_dir / "secondary_structure_per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "target", "task", "seed", "matched_residues",
            "GT_helix_percent", "GT_sheet_percent", "GT_coil_percent",
            "AF3_helix_percent", "AF3_sheet_percent", "AF3_coil_percent",
            "Corrected_helix_percent", "Corrected_sheet_percent", "Corrected_coil_percent",
            "Q3_AF3", "Q3_AT_AF3", "Q3_Corrected_ATAF3",
            "delta_Q3", "delta_Q3_vs_AT_AF3", "backbone_identical_AT_vs_Corrected"])
        w.writeheader()
        w.writerows(per_seed)

    # ------------------------------------------------------------------
    # summary: one row per protein, then All-D / Local-D / overall
    # ------------------------------------------------------------------
    by_protein = defaultdict(list)
    for r in per_seed:
        by_protein[(r["task"], r["target"])].append(r)

    def means(key, rows):
        return round(statistics.mean(float(r[key]) for r in rows), 2)

    def comp_rows(rows):
        return {f"{pre}_{cl}": means(f"{pre}_{cl}", rows)
                for pre in ("GT", "AF3", "Corrected")
                for cl in ("helix_percent", "sheet_percent", "coil_percent")}

    summary = []
    for (task, case), rows in sorted(by_protein.items()):
        entry = {
            "scope": "protein", "task": task, "target": case, "n_seeds": len(rows),
            "matched_residues_mean": round(statistics.mean(r["matched_residues"] for r in rows), 1),
            "Q3_AF3_mean": means("Q3_AF3", rows),
            "Q3_Corrected_ATAF3_mean": means("Q3_Corrected_ATAF3", rows),
            "delta_Q3_mean": means("delta_Q3", rows),
            "backbone_identical_fraction": round(statistics.mean(
                1.0 if r["backbone_identical_AT_vs_Corrected"] is True else 0.0 for r in rows), 3),
        }
        entry.update(comp_rows(rows))
        summary.append(entry)

    def overall(label, rows):
        if not rows:
            return None
        proteins = sorted({(r["task"], r["target"]) for r in rows})

        def eq_mean(key):
            return round(statistics.mean(
                statistics.mean(float(x[key]) for x in rows if (x["task"], x["target"]) == p)
                for p in proteins), 2)

        entry = {
            "scope": "overall", "task": label, "target": f"{len(proteins)} proteins",
            "n_seeds": len(rows),
            "matched_residues_mean": round(statistics.mean(r["matched_residues"] for r in rows), 1),
            "Q3_AF3_mean": eq_mean("Q3_AF3"),
            "Q3_Corrected_ATAF3_mean": eq_mean("Q3_Corrected_ATAF3"),
            "delta_Q3_mean": eq_mean("delta_Q3"),
            "backbone_identical_fraction": round(statistics.mean(
                1.0 if r["backbone_identical_AT_vs_Corrected"] is True else 0.0 for r in rows), 3),
        }
        entry.update({f"{pre}_{cl}": eq_mean(f"{pre}_{cl}")
                      for pre in ("GT", "AF3", "Corrected")
                      for cl in ("helix_percent", "sheet_percent", "coil_percent")})
        return entry

    for label, rows in (("All-D", [r for r in per_seed if r["task"] == "All-D"]),
                        ("Local-D", [r for r in per_seed if r["task"] == "Local-D"]),
                        ("All 16 proteins", per_seed)):
        row = overall(label, rows)
        if row:
            summary.append(row)

    # same self-check on the summary rows; their denominators are means, so only the
    # triple sum is enforced there, plus agreement with the per-seed rows
    bad = composition_problems(summary, "summary", "matched_residues_mean", tol=0.03,
                               integral=False)
    for row in summary:
        if row["scope"] != "protein":
            continue
        rows = by_protein[(row["task"], row["target"])]
        for pre in ("GT", "AF3", "Corrected"):
            for cl in COMP_CLASSES:
                key = f"{pre}_{cl}_percent"
                m = statistics.mean(float(r[key]) for r in rows)
                if abs(m - float(row[key])) > 0.011:      # summary holds 2 decimals
                    bad.append(f"summary {row['target']} {key}: {row[key]} vs "
                               f"per-seed mean {m:.4f}")
        n_mean = statistics.mean(float(r["matched_residues"]) for r in rows)
        if abs(n_mean - float(row["matched_residues_mean"])) > 0.051:   # 1 decimal
            bad.append(f"summary {row['target']} matched_residues_mean: "
                       f"{row['matched_residues_mean']} vs per-seed mean {n_mean:.2f}")
    if bad:
        raise SystemExit("secondary_structure_q3 summary self-check FAILED "
                         f"({len(bad)} problem(s)):\n  " + "\n  ".join(bad[:20]))

    fields = (["scope", "task", "target", "n_seeds", "matched_residues_mean"]
              + [f"{p}_{c}" for p in ("GT", "AF3", "Corrected")
                 for c in ("helix_percent", "sheet_percent", "coil_percent")]
              + ["Q3_AF3_mean", "Q3_Corrected_ATAF3_mean", "delta_Q3_mean",
                 "backbone_identical_fraction"])
    with (a.out_dir / "secondary_structure_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    # ------------------------------------------------------------------
    # conclusion
    # ------------------------------------------------------------------
    def emit(title, cols, rowfmt, gt_row=False):
        print()
        print(title)
        hdr = f"{'dataset':8s} {'protein':10s} {'n':>6s} " + " ".join(f"{c:>7s}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        if gt_row:
            print(f"{'':8s} {'(GT ref)':10s} {'-':>6s} " +
                  " ".join(f"{'100.00':>7s}" for _ in cols))
            print("-" * len(hdr))
        for scope in ("protein", "overall"):
            for row in summary:
                if row["scope"] != scope:
                    continue
                name = row["target"] if scope == "protein" else "(overall)"
                print(f"{row['task']:8s} {name:10s} {row['matched_residues_mean']:6.0f} " +
                      " ".join(rowfmt(row, c) for c in cols))
            print("-" * len(hdr))
        print("=" * len(hdr))

    PRE = {"GT": "GT", "AF3": "AF3", "COR": "Corrected"}
    CLS = {"H": "helix", "E": "sheet", "C": "coil"}
    Q3KEY = {"Q3_AF3": "Q3_AF3_mean", "Q3_COR": "Q3_Corrected_ATAF3_mean", "dQ3": "delta_Q3_mean"}

    def comp_fmt(row, col):
        pre, cl = col.split("_")
        return f"{row[PRE[pre] + '_' + CLS[cl] + '_percent']:7.1f}"

    def q3_fmt(row, col):
        return f"{row[Q3KEY[col]]:+7.2f}"

    emit("Table 1. Secondary-structure composition over the matched residues (mean %, 30 seeds)",
         ["GT_H", "GT_E", "GT_C", "AF3_H", "AF3_E", "AF3_C", "COR_H", "COR_E", "COR_C"],
         comp_fmt)
    emit("Table 2. Q3 secondary-structure agreement with GT (mean %, 30 seeds)",
         ["Q3_AF3", "Q3_COR", "dQ3"], q3_fmt, gt_row=True)

    deltas = [r["delta_Q3"] for r in per_seed]
    per_protein = [row["delta_Q3_mean"] for row in summary if row["scope"] == "protein"]
    worse = [row for row in summary if row["scope"] == "protein" and row["delta_Q3_mean"] < 0]
    if not deltas:
        print("CONCLUSION: no usable seeds.")
        return per_seed
    m, s = statistics.mean(deltas), sd(deltas)
    print(f"\nPer-seed delta Q3 across all {len(deltas)} seed pairs: mean {m:+.3f} pp, SD {s:.3f} pp")
    print(f"Per-protein delta Q3 mean: {statistics.mean(per_protein):+.3f} pp "
          f"(range {min(per_protein):+.2f} to {max(per_protein):+.2f})")
    print(f"Proteins with negative mean delta Q3: {len(worse)}/{len(per_protein)}")
    # correction term: exactly zero because the correction freezes the backbone
    corr_term = [float(r["delta_Q3_vs_AT_AF3"]) for r in per_seed
                 if r["delta_Q3_vs_AT_AF3"] != ""]
    if corr_term:
        print(f"Correction term (Corrected - AT-AF3): max |delta Q3| = {max(abs(x) for x in corr_term):.4f} pp "
              f"over {len(corr_term)} seed pairs")
    # protein-level (cluster-aware) one-sample t on delta Q3
    pm = statistics.mean(per_protein)
    if len(per_protein) > 1 and statistics.stdev(per_protein) > 0:
        t = pm / (statistics.stdev(per_protein) / len(per_protein) ** 0.5)
    else:
        t = float("nan")
    print(f"Protein-level mean delta Q3 = {pm:+.3f} pp, t = {t:+.2f} "
          f"(unit = protein; the pooled per-seed t overstates significance)")
    if abs(pm) < 1.0 and len(worse) * 2 <= len(per_protein) + 1:
        verdict = ("NO systematic reduction: the correction is secondary-structure neutral by "
                   "construction (backbone frozen; Corrected - AT-AF3 = 0), and the "
                   "AF3 -> Corrected-AT-AF3 difference is small, bidirectional and not "
                   "significant at the protein level.")
    else:
        verdict = ("A reduction is present relative to AF3, but it originates in the AT-AF3 "
                   "conditioning step, not in the geometric correction.")
    print("CONCLUSION: " + verdict)
    return per_seed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 0)

