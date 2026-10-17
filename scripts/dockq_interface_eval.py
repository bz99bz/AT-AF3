#!/usr/bin/env python
"""Per-interface DockQ v2 evaluation of the 16 AT-AF targets, with D-residue retention.

The references in ``canonical_pdb/*_seed_0_GT.pdb`` store every D-residue as a **HETATM**
record with its D-CCD name, while DockQ v2 reads polymer residues with
``parse_hetatms=False``.  The script therefore

1. writes a normalised DockQ input for both the reference and the prediction in which every
   **polymer** residue is an ``ATOM`` record with a standard residue name (D-CCD -> its L
   parent, other non-standard -> their parent or ALA).  Chain ids, residue numbers,
   insertion codes, atom names and coordinates are unchanged, so the intended chain
   identities and the D-residue geometry are retained; only the label DockQ does not parse
   is translated.  Chirality itself is *not* measured here - that is the signed-volume test
   of ``gt_metrics_suite.py``.  Residue coverage before and after normalisation is written
   to ``dockq_input_coverage.csv``.
2. enumerates **every relevant interface** of the reference (chain pair with a heavy-atom
   contact below DockQ's 10 A interface threshold) and runs DockQ once per
   (target, seed, condition) with an explicit ``--mapping``, so DockQ scores each interface
   of the mapped chain set separately and all per-interface values are kept next to the
   average (``best_result`` keyed by the native chain pair).
3. checks the enumeration per interface with a reference self-test
   (``DockQ(reference, reference)``, reported alongside the interface size ``nat_total``).
4. marks single-chain references and interfaces without a model counterpart as
   ``not applicable`` - never 0.
5. pairs the **same seed** across AF3 / AT-AF3 / corrected and reports the change per
   target *and per interface*, with the number of improved/worsened seeds, because a pooled
   mean can hide deterioration of an individual assembly.

Outputs (written under ``--out-dir``, default ``reports_master42_n30_ccdfix``):
  dockq_interface_per_seed.csv
  dockq_interface_summary.csv
  dockq_interface_paired.csv
  dockq_input_coverage.csv
  dockq_interface_report.md

Usage:
  phenix.python scripts/dockq_interface_eval.py --cases 8GQP_1 7W40_6 --max-seeds 2
  phenix.python scripts/dockq_interface_eval.py          # full 630-model sweep
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gt_metrics_suite import (  # noqa: E402
    ALL_D, CANON, COND_ORDER, COND_SUFFIX, DEFAULT_DOCKQ, LOCAL_D,
    discover, load_chains, pair_chains, target_chain_list)

CHIR = ROOT / "configs" / "target_chirality"
REPORTS = ROOT / "reports_master42_n30_ccdfix"
NA = "not applicable"
DOCKQ_INTERFACE_CUTOFF = 10.0      # DockQ constants.INTERFACE_THRESHOLD
DOCKQ_FNAT_CUTOFF = 5.0            # DockQ constants.FNAT_THRESHOLD

D_TO_L = {"DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS", "DGL": "GLU",
          "DGN": "GLN", "DSG": "ASN", "DSN": "SER", "DHI": "HIS", "DIL": "ILE",
          "DLE": "LEU", "DLY": "LYS", "MED": "MET", "DPN": "PHE", "DPR": "PRO",
          "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL", "DUN": "ALA",
          "DNP": "ALA", "DVA ": "VAL"}
STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
       "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
OTHER_PARENT = {"AIB": "ALA", "BF7": "ALA", "28J": "ALA", "HJV": "ALA", "ZAE": "ALA",
                "BAL": "ALA", "NLN": "ALA", "PG6": "ALA", "MLU": "LEU", "SAR": "GLY",
                "NLE": "LEU", "SEP": "SER", "TPO": "THR", "MLY": "LYS", "ORN": "LYS",
                "CSO": "CYS", "CME": "CYS", "MSE": "MET"}
BACKBONE = ("N", "CA", "C")


def normalised_name(resname: str) -> str:
    """Residue name DockQ can parse: D-CCD -> L parent, known non-standard -> parent."""
    rn = resname.strip().upper()
    if rn in D_TO_L:
        return D_TO_L[rn]
    if rn in STD:
        return rn
    return OTHER_PARENT.get(rn, "ALA")


# ---------------------------------------------------------------------------
#  PDB reading / normalisation / residue coverage
# ---------------------------------------------------------------------------

def _element(line: str, aname: str) -> str:
    el = line[76:78].strip().upper() if len(line) >= 78 else ""
    if el:
        return el
    return "".join(c for c in aname if c.isalpha())[:2].upper()


def read_polymer(path: Path, chains=None, only_atom_records: bool = False):
    """{chain: {res_key: {'name': str, 'atoms': {aname: (x, y, z, occ, elt)}}}}

    A residue counts as polymer when ``N``, ``CA`` and ``C`` are present - the same rule as
    ``gt_metrics_suite.load_chains``.  Hydrogens and the lower-occupancy duplicate are
    dropped, so the numbers here are directly comparable with the suite's residue counts.
    """
    out = {}
    for line in path.open(errors="replace"):
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM") or (only_atom_records and rec != "ATOM"):
            continue
        ch = line[21]
        if chains and ch not in chains:
            continue
        aname = line[12:16].strip()
        elt = _element(line, aname)
        if elt == "H":
            continue
        key = (line[22:27].strip(), line[26].strip())
        res = out.setdefault(ch, {}).setdefault(key, {"name": line[17:20].strip().upper(),
                                                      "atoms": {}})
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            occ = float(line[54:60] or 1.0)
        except ValueError:
            continue
        old = res["atoms"].get(aname)
        if old is None or occ > old[3]:
            res["atoms"][aname] = (xyz[0], xyz[1], xyz[2], occ, elt)
    keep = {}
    for ch, res in out.items():
        good = {k: v for k, v in res.items() if set(BACKBONE) <= set(v["atoms"])}
        if good:
            keep[ch] = good
    return keep


def write_normalised(src: Path, dst: Path, chains=None):
    """Write an ATOM-only, standard-residue-name copy of the polymer part of `src`.

    Returns {chain: {'residues': n, 'renamed': n, 'atoms': n}}.  Chain ids, residue
    numbers, insertion codes and coordinates are preserved (intended chain identities),
    so DockQ scores the intended interface with the intended residues.
    """
    res_all = read_polymer(src, chains)
    stats, serial = {}, 0
    with dst.open("w") as fh:
        for ch in sorted(res_all):
            n_ren, n_at = 0, 0

            def sort_key(k):
                return (int(k[0]) if k[0].lstrip("-").isdigit() else 0, k[1])

            for key in sorted(res_all[ch], key=sort_key):
                res = res_all[ch][key]
                name = normalised_name(res["name"])
                n_ren += int(name != res["name"])
                num = int(key[0]) if key[0].lstrip("-").isdigit() else 0
                for aname in sorted(res["atoms"]):
                    x, y, z, occ, elt = res["atoms"][aname]
                    serial += 1
                    a = aname if len(aname) >= 4 else f" {aname:<3s}"
                    fh.write(f"ATOM  {serial:5d} {a} {name:>3s} {ch:1s}{num:4d}"
                             f"{key[1]:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{0.0:6.2f}"
                             f"          {elt:>2s}\n")
                    n_at += 1
            stats[ch] = {"residues": len(res_all[ch]), "renamed": n_ren, "atoms": n_at}
        fh.write("END\n")
    return stats



def heavy_atom_contacts(chains: dict, a: str, b: str, cutoff: float):
    """(atom contacts < cutoff, contacting residue pairs, smallest distance) for two chains."""
    import numpy as np
    idx_a, pos_a = [], []
    for i, k in enumerate(sorted(chains[a])):
        for an, v in chains[a][k]["atoms"].items():
            idx_a.append(i)
            pos_a.append(v[:3])
    idx_b, pos_b = [], []
    for i, k in enumerate(sorted(chains[b])):
        for an, v in chains[b][k]["atoms"].items():
            idx_b.append(i)
            pos_b.append(v[:3])
    if not pos_a or not pos_b:
        return 0, 0, float("nan")
    A = np.asarray(pos_a, dtype=float)
    B = np.asarray(pos_b, dtype=float)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    sel = d < cutoff
    n_atom = int(sel.sum())
    if not n_atom:
        return 0, 0, float(d.min())
    ia = np.asarray(idx_a)[:, None].repeat(len(pos_b), 1)[sel]
    ib = np.asarray(idx_b)[None, :].repeat(len(pos_a), 0)[sel]
    n_pairs = len(set(zip(ia.tolist(), ib.tolist())))
    return n_atom, n_pairs, float(d[sel].min())


def native_interfaces(chains: dict, target_chains=None):
    """Every relevant reference interface: chain pairs with a heavy-atom contact below
    DockQ's 10 A interface threshold.  Keyed by the native chain pair, e.g. 'AB'.

    `role` marks whether the pair is an *intended* interface (at least one of the two chains
    is a curated target chain) or a contact between other components of the same file (for
    the All-D racemic entries this is the L-enantiomer crystal contact).
    """
    out = {}
    names = sorted(chains)
    tset = set(target_chains or [])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_atom, n_pairs, dmin = heavy_atom_contacts(chains, a, b, DOCKQ_INTERFACE_CUTOFF)
            if not n_atom:
                continue
            n5, p5, _ = heavy_atom_contacts(chains, a, b, DOCKQ_FNAT_CUTOFF)
            out[f"{a}{b}"] = {"chains": (a, b), "atom_contacts_10A": n_atom,
                              "residue_pairs_10A": n_pairs, "min_dist": dmin,
                              "atom_contacts_5A": n5, "residue_pairs_5A": p5,
                              "role": ("target interface" if not tset or a in tset or b in tset
                                       else "non-target contact")}
    return out


def run_dockq(binary: str, model: Path, native: Path, mapping: str, tag: str,
              workdir: Path, n_cpu: int = 1, timeout: int = 900):
    """DockQ with a pinned mapping -> ({'AB': {...}}, note).  Empty dict when it cannot score."""
    out_json = workdir / f"{tag}.json"
    if out_json.exists():
        out_json.unlink()
    cmd = [binary, str(model), str(native), "--short", "--json", str(out_json),
           "--n_cpu", str(n_cpu)]
    if mapping:
        cmd += ["--mapping", mapping]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - external tool
        return {}, f"dockq not run ({type(exc).__name__}: {exc})"
    if not out_json.exists():
        txt = (proc.stdout + proc.stderr).strip().splitlines()
        msg = next((ln for ln in txt if ln.startswith("ERROR")), txt[-1] if txt else "")
        return {}, msg.replace("ERROR:root:", "").strip()[:80]
    best = json.loads(out_json.read_text()).get("best_result") or {}
    if not best:
        return {}, "json without scorable interfaces"
    return best, ""


def chain_map_of(gt: dict, pred: dict, target_chains):
    """The benchmark's intended chain identity mapping: {native chain: model chain}."""
    out = {}
    for nchain, pairs in pair_chains(gt, pred, target_chains).items():
        if pairs:
            out[nchain] = pairs[0][1][0]        # the model chain this native chain maps to
    return out


def mapping_string(chain_map: dict) -> str:
    """DockQ --mapping MODELCHAINS:NATIVECHAINS (model order follows the native order)."""
    nat = [c for c in sorted(chain_map) if chain_map[c]]
    mod = [chain_map[c] for c in nat]
    return f"{''.join(mod)}:{''.join(nat)}"


# ---------------------------------------------------------------------------
#  the sweep
# ---------------------------------------------------------------------------

def iso(x, nd=4):
    """Numeric cell or '' (never 0 for an undefined quantity)."""
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else ""


def main():
    ap = argparse.ArgumentParser(description="Per-interface DockQ v2 evaluation (D-retaining)")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of target ids")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--dockq", default=DEFAULT_DOCKQ)
    ap.add_argument("--n-cpu", type=int, default=1)
    ap.add_argument("--coverage-only", action="store_true",
                    help="only rebuild dockq_input_coverage.csv (no DockQ runs)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the CSVs and the report from the existing per-seed CSV")
    args = ap.parse_args()

    if args.coverage_only:
        return coverage_only(args)
    if args.report_only:
        return report_only(args)

    conditions = args.conditions or COND_ORDER
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="dockq_iface_"))
    workdir.mkdir(parents=True, exist_ok=True)
    binary = args.dockq
    if not Path(binary).exists():
        print(f"warning: DockQ binary not found at {binary}")

    per_seed, coverage = [], []
    for dataset, cases, kind in (("All-D", ALL_D, "All_D"), ("Local-D", LOCAL_D, "Local_D")):
        for case in cases:
            if args.cases and case not in args.cases:
                continue
            gt_path = CANON / f"{kind}_{case}_seed_0_GT.pdb"
            gt_poly = read_polymer(gt_path)
            gt_visible = read_polymer(gt_path, only_atom_records=True)
            gt_chains_all = load_chains(gt_path)
            tchains_all = target_chain_list(case, dataset, gt_chains_all, CHIR)
            ifaces = native_interfaces(gt_poly, tchains_all)
            nat_norm = workdir / f"{kind}_{case}_GT_norm.pdb"
            write_normalised(gt_path, nat_norm)
            self_map = {c: c for c in sorted(gt_poly)}
            self_best, self_note = ({}, "no DockQ")
            if len(self_map) >= 2 and Path(binary).exists():
                self_best, self_note = run_dockq(binary, nat_norm, nat_norm,
                                                 mapping_string(self_map),
                                                 f"{kind}_{case}_self", workdir, args.n_cpu)
            for ch in sorted(gt_poly):
                coverage.append(dict(
                    dataset=dataset, target=case, role="reference", chain=ch,
                    residues=len(gt_poly[ch]),
                    dockq_visible_before=len(gt_visible.get(ch, {})),
                    residues_after=len(read_polymer(nat_norm, [ch]).get(ch, {})),
                    verdict=("ok" if len(gt_visible.get(ch, {})) == len(gt_poly[ch])
                             else "invisible to DockQ before normalisation")))
            print(f"[{dataset}] {case}: {len(ifaces)} reference interfaces "
                  f"{sorted(ifaces)}; self-check {self_note or 'ok'}", flush=True)
            gt = load_chains(gt_path)
            tchains = target_chain_list(case, dataset, gt, CHIR)
            files = discover(CANON, kind, case)
            seeds = sorted(s for s in files if isinstance(s, int))
            if args.max_seeds:
                seeds = seeds[:args.max_seeds]
            run_seeds(args, dataset, case, kind, files, seeds, conditions, gt_path, gt,
                      tchains, ifaces, self_best, nat_norm, binary, workdir, per_seed)
    write_outputs(args.out_dir, per_seed, coverage, conditions)
    print(f"rows: {len(per_seed)}; outputs in {args.out_dir}")
    return 0


def coverage_only(args):
    """Rebuild `dockq_input_coverage.csv` only (reference + every prediction; no DockQ run).

    For every file the normalised copy is actually written and re-read, so the
    "visible after" column is a measurement, not an assumption.
    """
    import tempfile
    rows = []
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="dockq_coverage_"))
    work.mkdir(parents=True, exist_ok=True)

    def coverage_file(path, dataset, case, role, extra=""):
        poly = read_polymer(path)
        vis = read_polymer(path, only_atom_records=True)
        norm = work / f"{case}_{path.stem}_norm.pdb"
        side = path.open(errors="replace")
        name = path.name
        side.close()
        write_normalised(path, norm)
        after = read_polymer(norm)
        for ch in sorted(poly):
            rows.append(dict(
                dataset=dataset, target=case, role=role, chain=ch, file=name,
                residues=len(poly[ch]), dockq_visible_before=len(vis.get(ch, {})),
                residues_after=len(after.get(ch, {})),
                verdict=("ok" if len(vis.get(ch, {})) == len(poly[ch]) else
                         "invisible to DockQ before normalisation (HETATM/D-CCD)")
                if after.get(ch) and len(after[ch]) == len(poly[ch]) else
                "NOT fully retained after normalisation"))

    for dataset, cases, kind in (("All-D", ALL_D, "All_D"), ("Local-D", LOCAL_D, "Local_D")):
        for case in cases:
            if args.cases and case not in args.cases:
                continue
            coverage_file(CANON / f"{kind}_{case}_seed_0_GT.pdb", dataset, case, "reference")
            files = discover(CANON, kind, case)
            seeds = sorted(s for s in files if isinstance(s, int))
            if args.max_seeds:
                seeds = seeds[:args.max_seeds]
            for seed in seeds:
                for cond in (args.conditions or COND_ORDER):
                    mpath = files[seed].get(COND_SUFFIX[cond])
                    if mpath is None or not Path(mpath).exists():
                        continue
                    coverage_file(mpath, dataset, case, f"prediction {cond}")
            print(f"  coverage recorded for {case}", flush=True)
    write_csv(args.out_dir / "dockq_input_coverage.csv", rows)
    print(f"wrote {args.out_dir}/dockq_input_coverage.csv ({len(rows)} rows)")
    return 0


def monomer_rows(per_seed, out_dir, coverage=None):
    """Rows for references that are a single chain: `not applicable`, never 0.

    The seed x condition grid comes from the published per-seed file of the metrics suite,
    the "single chain" fact from the reference rows of the coverage table.
    """
    suite_path = REPORTS / "gt_metrics_per_seed.csv"
    if coverage is None:
        coverage_path = out_dir / "dockq_input_coverage.csv"
        if not coverage_path.exists():
            return []
        coverage = list(csv.DictReader(coverage_path.open(newline="")))
    if not suite_path.exists():
        return []
    chains = collections.defaultdict(set)
    for r in coverage:
        if r.get("role") == "reference":
            chains[(r["dataset"], r["target"])].add(r["chain"])
    single = {k for k, v in chains.items() if len(v) <= 1}
    have = {(r["dataset"], r["target"], r["seed"], r["condition"]) for r in per_seed}
    rows = []
    for r in csv.DictReader(suite_path.open(newline="")):
        key = (r["dataset"], r["target"], r["seed"], r["condition"])
        if (r["dataset"], r["target"]) not in single or key in have:
            continue
        rows.append(dict(
            dataset=r["dataset"], target=r["target"], seed=r["seed"],
            condition=r["condition"], interface="-", native_chains="-", status=NA,
            dockq="", fnat="", iRMSD="", LRMSD="", nat_correct="", nat_total="",
            model_total="", fnonnat="", clashes="", len1="", len2="",
            native_atoms_5A="", native_atoms_10A="", native_res_pairs_5A="", min_dist="",
            interface_role="reference is a single chain", reference_selfcheck="",
            mapping="", model_chains="", note="reference has one chain - quaternary structure "
                                              "not applicable"))
    return rows


def report_only(args):
    """Rebuild the per-seed/summary/paired CSVs and the report from the existing per-seed CSV.

    Adds the `not applicable` rows of the single-chain references, so every monomer is
    represented explicitly.
    """
    src = args.out_dir / "dockq_interface_per_seed.csv"
    if not src.exists():
        print(f"missing {src}")
        return 1
    per_seed = list(csv.DictReader(src.open(newline="")))
    per_seed += monomer_rows(per_seed, args.out_dir)
    conditions = args.conditions or COND_ORDER
    write_outputs(args.out_dir, per_seed, list(csv.DictReader(
        (args.out_dir / "dockq_input_coverage.csv").open(newline=""))), conditions)
    print(f"rebuilt from {src.name}: {len(per_seed)} per-seed interface rows")
    return 0


def run_seeds(args, dataset, case, kind, files, seeds, conditions, gt_path, gt, tchains,
              ifaces, self_best, nat_norm, binary, workdir, per_seed):
    """One DockQ call per (seed, condition) with the pinned chain mapping -> all interfaces."""
    for n_done, seed in enumerate(seeds, start=1):
        for cond in conditions:
            suffix = COND_SUFFIX[cond]
            mpath = files[seed].get(suffix)
            if mpath is None or not Path(mpath).exists():
                continue
            pred = load_chains(mpath)
            cmap = chain_map_of(gt, pred, tchains)
            mod_norm = workdir / f"{kind}_{case}_{seed}_{suffix}_norm.pdb"
            write_normalised(mpath, mod_norm)
            pairs_mod = sorted(set(cmap.values()))
            if len(cmap) >= 2 and len(pairs_mod) >= 2 and Path(binary).exists():
                best, note = run_dockq(binary, mod_norm, nat_norm, mapping_string(cmap),
                                       f"{kind}_{case}_{seed}_{suffix}", workdir, args.n_cpu)
            else:
                best, note = {}, ("single-chain prediction - quaternary structure "
                                  "not applicable" if len(cmap) < 2
                                  else "no chain mapping to the reference")
            for key, info in sorted(ifaces.items()):
                a, b = info["chains"]
                covered = a in cmap and b in cmap and cmap[a] != cmap[b]
                res = best.get(key, {}) if covered else {}
                status = ("scored" if res else
                          "no interface found by DockQ" if covered else NA)
                per_seed.append(dict(
                    dataset=dataset, target=case, seed=seed, condition=cond,
                    interface=key, native_chains=f"{a}-{b}", status=status,
                    dockq=iso(res.get("DockQ")), fnat=iso(res.get("fnat"), 3),
                    iRMSD=iso(res.get("iRMSD"), 2), LRMSD=iso(res.get("LRMSD"), 2),
                    nat_correct=res.get("nat_correct", ""),
                    nat_total=res.get("nat_total", ""),
                    model_total=res.get("model_total", ""),
                    fnonnat=iso(res.get("fnonnat"), 3), clashes=res.get("clashes", ""),
                    len1=res.get("len1", ""), len2=res.get("len2", ""),
                    native_atoms_5A=info["atom_contacts_5A"],
                    native_atoms_10A=info["atom_contacts_10A"],
                    native_res_pairs_5A=info["residue_pairs_5A"],
                    interface_role=info.get("role", ""),
                    min_dist=iso(info["min_dist"], 2),
                    reference_selfcheck=(iso(self_best.get(key, {}).get("DockQ"))
                                         if key in self_best else ""),
                    mapping=mapping_string(cmap) if cmap else "",
                    model_chains=",".join(pairs_mod), note=note))
        if n_done % 5 == 0:
            print(f"   {case}: {n_done}/{len(seeds)} seeds ({len(per_seed)} rows)", flush=True)


def summarise(per_seed):
    """(dataset, target, interface, condition) -> mean/sd/min/max over the scored seeds."""
    by = collections.defaultdict(list)
    meta = {}
    for r in per_seed:
        key = (r["dataset"], r["target"], r["interface"], r["condition"])
        meta.setdefault((r["dataset"], r["target"], r["interface"]),
                        (r["native_chains"], r["native_atoms_5A"], r["native_atoms_10A"],
                         r["reference_selfcheck"]))
        if r["status"] == "scored" and r["dockq"]:
            by[key].append(float(r["dockq"]))
    rows = []
    for (ds, tgt, iface, cond), vals in sorted(by.items()):
        m = meta.get((ds, tgt, iface), ("", "", "", ""))
        rows.append(dict(dataset=ds, target=tgt, interface=iface, native_chains=m[0],
                         native_atoms_5A=m[1], native_atoms_10A=m[2],
                         reference_selfcheck=m[3], condition=cond,
                         n_seeds=len(vals), mean=iso(statistics.mean(vals)),
                         sd=iso(statistics.stdev(vals) if len(vals) > 1 else 0.0),
                         min=iso(min(vals)), max=iso(max(vals))))
    return rows



def paired_changes(per_seed):
    """Per target and per interface: the **same seed** before and after correction.

    DockQ is "higher is better", so a positive delta is an improvement.  Seeds are paired
    one-to-one (AF3 / AT-AF3 / corrected of the same seed), never pooled across seeds.
    """
    idx, meta = {}, {}
    for r in per_seed:
        idx[(r["dataset"], r["target"], r["seed"], r["interface"], r["condition"])] = (
            float(r["dockq"]) if r["status"] == "scored" and r["dockq"] else None)
        meta[(r["dataset"], r["target"], r["interface"])] = (
            r["native_chains"], r["native_atoms_5A"], r["reference_selfcheck"])
    combos = [("AT-AF3 - AF3", "AF3", "AT-AF3"),
              ("Corrected AT-AF3 - AF3", "AF3", "Corrected AT-AF3"),
              ("Corrected AT-AF3 - AT-AF3", "AT-AF3", "Corrected AT-AF3")]
    rows = []
    for ds, tgt, iface in sorted({(k[0], k[1], k[3]) for k in idx}):
        seeds = sorted({k[2] for k in idx if k[0] == ds and k[1] == tgt and k[3] == iface})
        for label, base, other in combos:
            deltas, bvals, ovals = [], [], []
            for s in seeds:
                a = idx.get((ds, tgt, s, iface, base))
                b = idx.get((ds, tgt, s, iface, other))
                if a is None or b is None:
                    continue
                deltas.append(b - a)
                bvals.append(a)
                ovals.append(b)
            if not deltas:
                continue
            m = meta.get((ds, tgt, iface), ("", "", ""))
            rows.append(dict(
                dataset=ds, target=tgt, interface=iface, native_chains=m[0],
                native_atoms_5A=m[1], reference_selfcheck=m[2], comparison=label,
                n_paired_seeds=len(deltas),
                mean_base=iso(statistics.mean(bvals)), mean_other=iso(statistics.mean(ovals)),
                mean_delta=iso(statistics.mean(deltas)),
                sd_delta=iso(statistics.stdev(deltas) if len(deltas) > 1 else 0.0),
                min_delta=iso(min(deltas)), max_delta=iso(max(deltas)),
                n_improved=sum(1 for d in deltas if d > 1e-9),
                n_worsened=sum(1 for d in deltas if d < -1e-9),
                n_unchanged=sum(1 for d in deltas if abs(d) <= 1e-9)))
    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)



def markdown_report(per_seed, summary, paired, coverage, conditions):
    L = []
    A = L.append
    A("# Per-interface DockQ v2 evaluation (D-residues retained)")
    A("")
    A("Generated by `scripts/dockq_interface_eval.py`.  Every reference interface is "
      "evaluated separately and the per-interface values are kept next to the average.  The "
      "AF3 / AT-AF3 / corrected predictions of the **same seed** are paired, and the change "
      "is reported per target and per interface, because a pooled mean can hide the "
      "deterioration of an individual assembly.")
    A("")
    A("A positive delta is an **improvement** (DockQ is higher-is-better).  "
      "`not applicable` means the prediction has no counterpart chain for that reference "
      "interface (single-chain prediction): it is never counted as 0.")
    A("")
    A("## 1. Reference residue coverage")
    A("")
    A("DockQ v2 reads with `parse_hetatms=False`, and every D-residue of the canonical "
      "references is written as a `HETATM` record, so it is invisible to DockQ.  The "
      "normalised input used here writes every polymer residue as `ATOM` with a standard "
      "residue name (D-CCD -> its L parent); chain id, residue number, insertion code and "
      "the coordinates are unchanged, so the intended chain identities and the D-residue "
      "geometry are retained.  Chirality itself is scored by the signed-volume test of "
      "`gt_metrics_suite.py`, not here.")
    A("")
    A("| Target | Chain | Residues in reference | Visible to DockQ before | Visible after | Verdict |")
    A("|---|---|---|---|---|---|")
    for r in coverage:
        if r["role"] == "reference":
            A(f"| {r['target']} | {r['chain']} | {r['residues']} | {r['dockq_visible_before']} | "
              f"{r['residues_after']} | {r['verdict']} |")
    A("")
    A("## 2. Reference interfaces evaluated (DockQ reference against itself)")
    A("")
    A("`Role` = `target interface` when at least one chain of the pair is a curated target "
      "chain (the All-D configs pin one or two chains, for the Local-D cases all chains of "
      "the selected assembly are target), otherwise `non-target contact`: a packing contact "
      "between the other components of the same reference file (for the racemic All-D "
      "entries this is an L-enantiomer contact, not the intended target interface).")
    A("")
    A("| Target | Interface | Role | Native contacts < 5 A | Native contacts < 10 A | DockQ(reference, reference) |")
    A("|---|---|---|---|---|---|")
    seen = set()
    for r in per_seed:
        if (r["target"], r["interface"]) in seen or r["interface"] == "-":
            continue
        seen.add((r["target"], r["interface"]))
        A(f"| {r['target']} | {r['native_chains']} | {r.get('interface_role', '')} | "
          f"{r['native_atoms_5A']} | {r['native_atoms_10A']} | "
          f"{r['reference_selfcheck'] or 'none found by DockQ'} |")
    A("")
    A("## 3. Per-target results, per interface (mean +/- SD over the scored seeds)")
    A("")
    A("Targets whose reference is a single chain have no interface to score and are listed "
      "as `not applicable` in section 5.")
    A("")
    A("| Target | Tier | Interface | Seeds | AF3 | AT-AF3 | Corrected | Paired delta (corrected - AF3) | Improved / worsened seeds |")
    A("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["dataset"], r["target"], r["interface"]) for r in per_seed
                   if r["interface"] != "-"})
    for ds, tgt, iface in keys:
        cells, nseeds = {}, 0
        for cond in conditions:
            vals = [float(r["dockq"]) for r in per_seed
                    if r["dataset"] == ds and r["target"] == tgt and r["interface"] == iface
                    and r["condition"] == cond and r["status"] == "scored" and r["dockq"]]
            nseeds = max(nseeds, len(vals))
            cells[cond] = (f"{statistics.mean(vals):.3f} +/- "
                           f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.3f}"
                           f" (n={len(vals)})") if vals else NA
        names = sorted({r["native_chains"] for r in per_seed
                        if r["dataset"] == ds and r["target"] == tgt and r["interface"] == iface})
        pr = [p for p in paired if p["dataset"] == ds and p["target"] == tgt
              and p["interface"] == iface and p["comparison"] == "Corrected AT-AF3 - AF3"]
        dcell = (f"{pr[0]['mean_delta']} +/- {pr[0]['sd_delta']} "
                 f"(range {pr[0]['min_delta']}..{pr[0]['max_delta']})") if pr else NA
        cnt = f"{pr[0]['n_improved']} / {pr[0]['n_worsened']}" if pr else NA
        A(f"| {tgt} | {ds} | {names[0] if names else '-'} | {nseeds} | {cells.get('AF3')} | "
          f"{cells.get('AT-AF3')} | {cells.get('Corrected AT-AF3')} | {dcell} | {cnt} |")
    A("")
    A("## 4. Per-target change - interfaces reported separately, never pooled")
    A("")
    A("`Mean delta (unweighted)` averages the per-interface deltas of that target. "
      "`Worst interface` shows that an average of a whole target can still hide one "
      "assembly that got worse.")
    A("")
    A("| Target | Tier | Interfaces scored | Interfaces worsening | Mean delta (unweighted) | Worst interface | Worst delta |")
    A("|---|---|---|---|---|---|---|")
    for ds, tgt in sorted({(r["dataset"], r["target"]) for r in per_seed}):
        pr = [p for p in paired if p["dataset"] == ds and p["target"] == tgt
              and p["comparison"] == "Corrected AT-AF3 - AF3"]
        if not pr:
            A(f"| {tgt} | {ds} | 0 | - | {NA} | - | - |")
            continue
        deltas = [(float(p["mean_delta"]), p["interface"]) for p in pr if p["mean_delta"]]
        worse = [p for p in pr if p["mean_delta"] and float(p["mean_delta"]) < -1e-9]
        worst = min(deltas) if deltas else (float("nan"), "-")
        A(f"| {tgt} | {ds} | {len(pr)} | {len(worse)} | "
          f"{statistics.mean([d for d, _ in deltas]):+.4f} | {worst[1]} | {worst[0]:+.4f} |")
    A("")
    A("## 5. Not-applicable entries (never 0)")
    A("")
    A("| Target | Interface | Status | Seeds x conditions |")
    A("|---|---|---|---|")
    na = collections.Counter()
    for r in per_seed:
        if r["status"] != "scored":
            na[(r["target"], r["interface"], r["status"])] += 1
    for (tgt, iface, status), n in sorted(na.items()):
        A(f"| {tgt} | {iface} | {status} | {n} |")
    A("")
    A("## 6. Method and limits")
    A("")
    A("* Interfaces: every chain pair of the reference with a heavy-atom contact below "
      "DockQ's 10 A interface threshold.  Each pair is scored separately by DockQ inside one "
      "run with an explicit `--mapping`, so `best_result` carries one entry per interface; "
      "no interface is averaged away.")
    A("* Chain identities: taken from `gt_metrics_suite.pair_chains`, i.e. the same "
      "sequence-based mapping (with the All-D target-chain pinning) that the fold metrics "
      "use, so the intended interface is the one scored.")
    A("* D-residues: retained in the coordinates; only their labels are translated so DockQ "
      "can parse them.  Handedness is therefore **not** measured by DockQ - the signed-volume "
      "chirality accuracy of `gt_metrics_suite.py` remains the chirality metric.")
    A("* Reference self-check: `DockQ(reference, reference)` is reported for every interface "
      "listed above; a value other than 1.0000 indicates a problem in the interface "
      "enumeration or the normalisation.")
    A("* Seeds: only the seeds that exist for both conditions of a comparison are paired, and "
      "the number of paired seeds is reported (`dockq_interface_paired.csv`).")
    A("* Status vocabulary: `scored` (DockQ produced a value for that interface), "
      "`no interface found by DockQ` (the model contains both chains but does not form that "
      "reference interface - not a 0 and not evaluable), `not applicable` (the prediction has "
      "no counterpart chain, i.e. a single-chain prediction, or the reference itself is a "
      "monomer).")
    A("* The geometric enumeration uses DockQ's 10 A interface threshold, so a pair with only "
      "long-range contacts is listed but may still be rejected by DockQ's own interface "
      "definition; such pairs show `none found by DockQ` in section 2.")
    return "\n".join(L) + "\n"


def write_outputs(out_dir: Path, per_seed, coverage, conditions):
    per_seed = per_seed + monomer_rows(per_seed, out_dir, coverage)
    summary = summarise(per_seed)
    paired = paired_changes(per_seed)
    write_csv(out_dir / "dockq_interface_per_seed.csv", per_seed)
    write_csv(out_dir / "dockq_interface_summary.csv", summary)
    write_csv(out_dir / "dockq_interface_paired.csv", paired)
    write_csv(out_dir / "dockq_input_coverage.csv", coverage)
    (out_dir / "dockq_interface_report.md").write_text(
        markdown_report(per_seed, summary, paired, coverage, conditions))
    return summary, paired


if __name__ == "__main__":
    sys.exit(main())

