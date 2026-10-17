#!/usr/bin/env python
"""Quaternary-structure inventory for the 16 AT-AF benchmark targets.

Question answered: does each benchmark protein have quaternary structure, and how
much of it does the benchmark actually compare?

Every fact in the table comes from an artefact that already exists in the
repository - nothing is re-derived or guessed:

  1. ``data/gt_reference/{all_d,local_d}/<PDB>.cif`` - the deposited mmCIF of
     each reference entry.  ``_pdbx_struct_assembly.oligomeric_details`` /
     ``.oligomeric_count`` is the author/software annotation of the biological
     assembly, i.e. the authoritative answer to "is this protein an oligomer".
  2. ``configs/at_af_7.json`` + ``configs/target_chirality/*.json`` - the curated
     All-D case metadata (entity, ``pdb_chain_context.asym_chains``, copies) and
     the chain the All-D target is pinned to.
  3. ``data/local_d/local_d_curated_v1_1/{06,22,23}_*.csv`` - the curated
     Local-D case/component/assembly manifests (assembly used, target chain,
     component entities and stoichiometry).
  4. ``results/gt_comparison_phenix_ccdfix/canonical_pdb/*_seed_0_GT.pdb`` - the
     reference file the benchmark actually scores: its chain inventory and the
     inter-chain C-alpha contacts are counted here from the coordinates.
  5. ``reports_master42_n30_ccdfix/gt_metrics_summary.csv`` - how many chains the
     AF3 / AT-AF3 / corrected predictions contain (``n_chains``).
  6. DockQ (installed external tool, same call as ``gt_metrics_suite.run_dockq``)
     is run on each reference **against itself**; ``1.0000`` means the reference
     scope contains at least one scorable protein-protein interface, a failure
     means DockQ could not build an interface from that file at all.

Output (both regenerated, nothing else touched):
  reports_master42_n30_ccdfix/quaternary_structure_table.md
  reports_master42_n30_ccdfix/quaternary_structure_table.csv

Usage:
  phenix.python scripts/quaternary_structure_table.py            # full run (16 DockQ calls)
  phenix.python scripts/quaternary_structure_table.py --no-dockq # skip the DockQ self-check
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gt_metrics_suite import DEFAULT_DOCKQ, load_chains, run_dockq  # noqa: E402
from dockq_interface_eval import write_normalised  # noqa: E402

CIF_DIR = ROOT / "data" / "gt_reference"
CANON = ROOT / "results" / "gt_comparison_phenix_ccdfix" / "canonical_pdb"
CFG = ROOT / "configs"
CHIR = CFG / "target_chirality"
LD_MAN = ROOT / "data" / "local_d" / "local_d_curated_v1_1"
REPORTS = ROOT / "reports_master42_n30_ccdfix"

ALL_D = ["7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"]
LOCAL_D = ["7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1",
           "8U78_1", "8UM9_1", "9BIC_1"]
TIERS = [("All-D", ALL_D, "all_d", "All_D"), ("Local-D", LOCAL_D, "local_d", "Local_D")]
CONTACT = 8.0            # C-alpha - C-alpha distance that counts as an inter-chain contact


def assembly_rows(text: str) -> list:
    """``_pdbx_struct_assembly`` rows, from the loop *or* the key/value mmCIF form."""
    rows = []
    m = re.search(r"loop_\s*\n((?:_pdbx_struct_assembly\.\w+\s*\n)+)((?:[^\n#]*\n)+)", text)
    if m:
        cols = re.findall(r"_pdbx_struct_assembly\.(\w+)", m.group(1))
        for line in m.group(2).strip().splitlines():
            vals = line.split()
            if len(vals) == len(cols):
                rows.append(dict(zip(cols, vals)))
        if rows:
            return rows
    cur = None
    for key, val in re.findall(r"^_pdbx_struct_assembly\.(\w+)\s+([^\s#]+)\s*$", text, re.M):
        if key == "id":
            if cur:
                rows.append(cur)
            cur = {}
        if cur is not None:
            cur[key] = val
    if cur:
        rows.append(cur)
    return rows


def entity_titles(text: str) -> dict:
    """entity_id -> one-line description (from ``_entity.pdbx_description``)."""
    out = {}
    m = re.search(r"loop_\s*\n((?:_entity\.\w+\s*\n)+)((?:[^\n#]*\n)+)", text)
    if not m:
        return out
    cols = re.findall(r"_entity\.(\w+)", m.group(1))
    if "id" not in cols or "pdbx_description" not in cols:
        return out
    for line in m.group(2).strip().splitlines():
        vals = line.split()
        if len(vals) == len(cols):
            d = dict(zip(cols, vals))
            out[d["id"]] = d["pdbx_description"]
    return out


def gt_chains(path: Path):
    """chain -> (n_residues, [CA coordinates]) for the canonical reference file.

    Uses the suite's own residue convention (`gt_metrics_suite.load_chains`: heavy atoms,
    first model of an NMR/nmr-style file, residues with N/CA/C), so the chain sizes here are
    directly comparable with `n_gt_res` of `gt_metrics_summary.csv`.
    """
    out = {}
    for c, ch in load_chains(path).items():
        ca = [tuple(float(v) for v in ch["xyz"][k]["CA"]) for k in ch["order"]
              if "CA" in ch["xyz"][k]]
        if ca:
            out[c] = (len(ch["order"]), ca)
    return out


def interface_evidence(chains: dict, max_pairs: int = 4):
    """(n_contacts, [(a, b, min_dist, n_contacts)], longest chain id).

    Every chain pair with at least one C-alpha...C-alpha contact below `CONTACT` A is
    listed: the chains of the reference scope are only in contact if the deposited
    assembly really holds them together in that coordinate file.
    """
    names = sorted(chains)
    pairs, total = [], 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = chains[names[i]][1], chains[names[j]][1]
            best, n = math.inf, 0
            for p in a:
                for q in b:
                    d = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)
                    if d < best:
                        best = d
                    if d < CONTACT:
                        n += 1
            if n:
                total += n
                pairs.append((names[i], names[j], best, n))
    pairs.sort(key=lambda t: (-t[3], t[2]))
    return total, pairs[:max_pairs], max(names, key=lambda c: chains[c][0]) if names else ""



def load_all_d():
    """case -> curated All-D metadata from configs/at_af_7.json + the chirality configs."""
    data = json.loads((CFG / "at_af_7.json").read_text())
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    out = {}
    for c in cases:
        cid = c.get("pdb_entity_id")
        if cid not in ALL_D:
            continue
        ctx = c.get("pdb_chain_context") or {}
        cfgp = CHIR / f"{cid}.json"
        tgt, mode = [], ""
        if cfgp.exists():
            cc = json.loads(cfgp.read_text())
            tgt = sorted((cc.get("chains") or {}).keys())
            mode = str(cc.get("mode", ""))
        out[cid] = dict(
            pdb=c.get("pdb_id"), entity=c.get("entity_id"), name=c.get("protein_name"),
            case_type=c.get("case_type"), entry_title=c.get("entry_title"),
            asym=ctx.get("asym_chains") or [], copies=ctx.get("observed_entity_copy_count"),
            target_chains=tgt, mode=mode,
            assembly_id="not recorded (All-D curated per chain)",
            assembly_reason="All-D cases are curated per target chain, not per assembly",
            entities={}, n_d_sites=None)
    return out


def load_local_d():
    """case -> curated Local-D metadata from the 06/22/23 manifests."""
    def rows(name):
        with (LD_MAN / name).open(newline="") as fh:
            return list(csv.DictReader(fh))

    cases = {r["case_id_normalized"]: r for r in rows("06_local_d_case_manifest.csv")}
    audit = {r["case_id"]: r for r in rows("23_assembly_selection_audit.csv")}
    comps = {}
    for r in rows("22_complex_component_manifest.csv"):
        if r["case_id"] in LOCAL_D:
            comps.setdefault(r["case_id"], []).append(r)
    out = {}
    for cid in LOCAL_D:
        r, a = cases.get(cid, {}), audit.get(cid, {})
        cc = comps.get(cid, [])
        ents = {}
        for c in cc:
            ents.setdefault(c["entity_id"], []).append(c["asym_id"])
        out[cid] = dict(
            pdb=r.get("pdb_id_normalized"), entity=r.get("entity_id"),
            name=r.get("entity_description"),
            case_type="local_D (L protein with site-specific D residues)",
            entry_title=r.get("entry_title"), asym=(r.get("asym_chains_raw") or "").split(","),
            copies=len(cc) or None, entities=ents,
            target_chains=[c for c in (a.get("target_chain_id") or "").split(",") if c],
            assembly_id=a.get("assembly_id"), assembly_reason=a.get("assembly_selection_reason"),
            n_d_sites=r.get("target_d_residue_count"))
    return out


def load_model_chains():
    """(dataset, target) -> {condition: chains in the prediction} from the published summary."""
    out = {}
    with (REPORTS / "gt_metrics_summary.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scope"] != "protein":
                continue
            out.setdefault((r["dataset"], r["target"]), {})[r["condition"]] = int(float(r["n_chains"]))
    return out



# ---------------------------------------------------------------------------
#  notes / verdicts that are judgement calls, kept in one place
# ---------------------------------------------------------------------------

NOTES = {
    "7P5R_2": "Racemic crystal, author annotation monomeric: the D target (C) and the second D copy (D) "
              "are crystal contacts, not a biological dimer.",
    "7QDI_1": "De novo D-protein, 8 copies in the asymmetric unit; the AF3 model is one chain, so the "
              "octamer is not part of the comparison.",
    "8AVR_2": "Racemic crystal (B,D = D copies; A,C = L enantiomer), author annotation monomeric.",
    "8BFD_2": "Racemic crystal with one D and one L copy; author annotation monomeric.",
    "8GQP_1": "Heterochiral complex (D-binder + L-target) and the only All-D target whose AF3 model is "
              "multi-chain.  The intended interface is A-B (the D binder against the D copy the "
              "normalisation reveals); DockQ's default parser skipped the HETATM D-residues, which is "
              "why the model-side DockQ used to read n/a here - see `dockq_interface_report.md`.",
    "8JH6_1": "The deposited dimer is the heterochiral D+L racemic pair, i.e. a racemic-crystallisation "
              "artifact rather than a D-D assembly; read it as a single-chain reference.",
    "8P6Q_2": "Racemic crystal, one D and one L copy; the D target has no biological partner in the "
              "entry, so read it as a single-chain reference.",
    "7OOJ_1": "Ubiquitin is monomeric; the two copies of the asymmetric unit are in contact, and the "
              "benchmark scores both of them.",
    "7P0A_3": "2:2:2 hetero-complex (heavy chain + beta-2-microglobulin + 9-residue peptide); the "
              "targeted local-D site lies in the peptide (C,F).",
    "7P0T_3": "Same 2:2:2 layout as 7P0A_3 (peptide = gp33-41).",
    "7W40_6": "Cryo-EM receptor/G-protein complex; the targeted local-D site is the bombesin-like chain F.",
    "8HR3_1": "13-residue D/L chimeric peptide, monomeric in the entry; single-chain comparison only.",
    "8HR4_1": "13-residue D/L chimeric peptide, monomeric in the entry; single-chain comparison only.",
    "8U78_1": "11-residue teixobactin analogue, 8 copies in the asymmetric unit (author annotation "
              "octameric); all 8 copies are in the benchmark scope.",
    "8UM9_1": "Immunoglobulin G-binding protein G B-domain, monomeric; single-chain comparison only.",
    "9BIC_1": "8-residue clovibactin analogue; the deposited assembly is a 16-mer (PISA) while the "
              "benchmark scope holds the 8 copies of the asymmetric unit.",
}


def build_rows(args):
    """One dict per target with every column of the table."""
    all_d, local_d = load_all_d(), load_local_d()
    model = load_model_chains()
    rows = []
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="quat_tbl_"))
    workdir.mkdir(parents=True, exist_ok=True)
    for tier, cases, cif_sub, kind in TIERS:
        meta = all_d if tier == "All-D" else local_d
        for case in cases:
            m = meta.get(case, {})
            pdb = m.get("pdb") or case.split("_")[0]
            cif = CIF_DIR / cif_sub / f"{pdb}.cif"
            text = cif.read_text(errors="ignore") if cif.exists() else ""
            asm = assembly_rows(text)
            details = [a.get("oligomeric_details", "?") for a in asm]
            counts = [a.get("oligomeric_count", "?") for a in asm]
            method = sorted({a.get("method_details", "?") for a in asm
                             if a.get("method_details") not in (None, "?")})
            dep = details[0] if details else "not annotated"
            # a reference "has quaternary structure" when the deposited assembly is not a monomer
            quat = "No" if dep.startswith("monomer") else ("Yes" if details else "?")
            gt_path = CANON / f"{kind}_{case}_seed_0_GT.pdb"
            chains = gt_chains(gt_path) if gt_path.exists() else {}
            n_contacts, pairs, _longest = interface_evidence(chains)
            iface = ("none - single chain" if len(chains) < 2 else
                     "none found" if n_contacts == 0 else
                     "; ".join(f"{a}-{b} {d:.1f} A ({n} CA)" for a, b, d, n in pairs))
            if args.no_dockq:
                dq = "not run"
            elif len(chains) >= 2 and Path(DEFAULT_DOCKQ).exists():
                # DockQ parses with `parse_hetatms=False` and the canonical references write every
                # D-residue as HETATM, so an un-normalised self-check would report DockQ's parser
                # rather than the reference (8JH6_1 / 8P6Q_2 used to read "none found" for that
                # reason).  The self-check therefore runs on the normalised copy - same chain ids,
                # residue numbers, atom names and coordinates, only the record type and the D-CCD
                # names translated.  Authoritative per-interface version:
                # `dockq_interface_report.md` section 2.
                norm = workdir / f"{case}_norm.pdb"
                write_normalised(gt_path, norm)
                summary, _note = run_dockq(DEFAULT_DOCKQ, norm, norm, workdir, f"quat_{case}")
                dq = f"{summary['dockq_mean']:.4f}" if summary else "none found"
            else:
                dq = "n/a (<2 chains)"
            mn = model.get((tier, case), {})
            af3, cor = mn.get("AF3"), mn.get("Corrected AT-AF3")
            rows.append(dict(
                target=case, tier=tier, pdb=pdb, entity=m.get("entity"),
                name=(m.get("name") or "?").strip()[:58],
                case_type=m.get("case_type", ""),
                deposited_assembly=dep, assembly_count=len(asm) or "?",
                assembly_ids=";".join(a.get("id", "?") for a in asm) or "?",
                assembly_method=",".join(method) if method else "author-defined",
                oligomeric_count=counts[0] if counts else "?",
                quaternary=quat,
                assembly_used=(f"{m.get('assembly_id')} ({m.get('assembly_reason')})"
                               if m.get("assembly_id") else "n/a"),
                entry_asym=",".join(str(c) for c in (m.get("asym") or [])),
                target_chains=",".join(m.get("target_chains") or []),
                n_entry_copies=m.get("copies", "?"),
                gt_chains=",".join(sorted(chains)),
                n_gt_chains=len(chains),
                gt_chain_sizes=";".join(f"{c}:{chains[c][0]}" for c in sorted(chains)),
                iface=iface, n_contacts=n_contacts, dockq_gt=dq,
                model_chains=af3, model_chains_cor=cor,
                compared_multi=("Yes - the prediction itself is multi-chain" if (af3 or 0) > 1
                                else "No - the prediction is a single chain"),
                note=NOTES.get(case, "")))
    return rows



COLS = ["Target", "Tier", "Reference entry", "Entity description", "Deposited assembly",
        "Quaternary structure", "Reference scope chains", "AF3 model chains",
        "Interface in the reference scope (CA-CA)", "DockQ (reference vs itself)",
        "Quaternary compared by the benchmark?", "Note"]


def md_row(r):
    """One markdown table row (cells kept short so the table stays readable)."""
    ent = f"{r['pdb']} entity {r['entity']}" if r["entity"] else str(r["pdb"])
    dep = f"{r['deposited_assembly']} (n={r['oligomeric_count']}"
    dep += f", {r['assembly_count']} assemblies" if str(r["assembly_count"]) not in ("1", "?") else ""
    dep += ")"
    gt = (f"{r['gt_chains']} ({r['gt_chain_sizes']})" if r["gt_chains"] else "no reference file")
    mod = f"{r['model_chains']}" if r["model_chains"] is not None else "?"
    if r["model_chains_cor"] not in (None, r["model_chains"]):
        mod += f" / {r['model_chains_cor']} corrected"
    return ("| " + " | ".join([
        r["target"], r["tier"], ent, r["name"], dep, "**" + r["quaternary"] + "**", gt, str(mod),
        r["iface"], r["dockq_gt"], r["compared_multi"], r["note"],
    ]) + " |")


def write_tables(rows, out_md: Path, out_csv: Path, dockq_used: bool):
    yes = [r for r in rows if r["quaternary"] == "Yes"]
    multi = [r for r in rows if (r["model_chains"] or 0) > 1]
    dq = [r for r in rows if re.fullmatch(r"\d+\.\d+", str(r["dockq_gt"]))]
    lines = []
    A = lines.append
    A("# Quaternary structure of the 16 AT-AF benchmark targets")
    A("")
    A("One row per benchmark protein.  Everything is read from artefacts that already exist "
      "in the repository (deposited mmCIF, curation manifests, canonical reference files, "
      "published metric summary); no plotted value is transcribed by hand.")
    A("")
    A("Definitions")
    A("")
    A("* **Deposited assembly** - `_pdbx_struct_assembly.oligomeric_details` / `.oligomeric_count` "
      "of the deposited entry (`data/gt_reference/{all_d,local_d}/<PDB>.cif`), i.e. the "
      "author/software annotation of the biological assembly.")
    A("* **Quaternary structure** - *Yes* when that annotation is not `monomeric`, i.e. the entry "
      "deposits the protein as an oligomer or as a complex; *No* when the biological assembly is a "
      "single chain.")
    A("* **Reference scope chains** - the polymer chains of the reference file the benchmark "
      "actually scores (`results/gt_comparison_phenix_ccdfix/canonical_pdb/*_seed_0_GT.pdb`), with "
      "their C-alpha counts; counted from the coordinates, not from the manifests.")
    A("* **AF3 model chains** - chains in the prediction (`n_chains` of the AF3 row in "
      "`gt_metrics_summary.csv`, 30 seeds; the corrected condition is shown when it differs).")
    A("* **Interface in the reference scope** - every inter-chain C-alpha...C-alpha contact below "
      "8 A in that reference file, as `<chains> <min distance> (<number of contacts>)`.")
    A("* **DockQ (reference vs itself)** - DockQ run on the reference against itself, **on the "
      "D-residue-normalised copy** of the reference (DockQ skips `HETATM` records by default): "
      "`1.0000` proves that the file holds at least one scorable protein-protein interface, "
      "`none found` means DockQ could not build an interface from that file, `n/a (<2 chains)` means "
      "the reference scope is a single chain.  This column counts *all* interfaces of the file, "
      "including racemic D+L pairs that are not biological partners - the per-interface breakdown "
      "with the intended mapping is `dockq_interface_report.md`.")
    A("")
    A("| " + " | ".join(COLS) + " |")
    A("|" + "|".join(["---"] * len(COLS)) + "|")
    for r in rows:
        A(md_row(r))
    A("")
    A(f"**{len(yes)} of {len(rows)} references have quaternary structure** "
      f"(All-D {sum(1 for r in yes if r['tier'] == 'All-D')}/7, "
      f"Local-D {sum(1 for r in yes if r['tier'] == 'Local-D')}/9); "
      f"{len(multi)} of {len(rows)} AF3 predictions are themselves multi-chain "
      f"({', '.join(r['target'] for r in multi)}); a scorable interface exists in "
      f"{len(dq)} of {len(rows)} reference scopes.")
    A("")
    with out_md.open("w") as fh:
        fh.write("\n".join(lines) + "\n")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(yes), len(multi), len(dq)




def load_dockq_published():
    """(dataset, target) -> {condition: published dockq_mean} for the multi-chain cases."""
    out = {}
    with (REPORTS / "gt_metrics_summary.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scope"] != "protein":
                continue
            if r.get("dockq_mean_mean"):
                out.setdefault((r["dataset"], r["target"]), {})[r["condition"]] = float(r["dockq_mean_mean"])
    return out


def append_analysis(out_md: Path, rows):
    """Second half of the note: what the inventory means for the published metrics."""
    dq = load_dockq_published()
    multi = [r for r in rows if (r["model_chains"] or 0) > 1]
    single = [r for r in rows if (r["model_chains"] or 0) <= 1]
    lines = []
    A = lines.append
    A("")
    A("## What this implies for the metric tables")
    A("")
    A(f"1. **{len(single)} of {len(rows)} targets are single-chain comparisons.** For "
      f"{', '.join(r['target'] for r in single)} the AF3 prediction contains one chain, so the "
      "benchmark can only test the fold (TM-score, C-alpha RMSD, lDDT, chirality) and DockQ is "
      "undefined - `n/a`, never zero.  The quaternary structure of those references exists in the "
      "deposited entry but is not part of the comparison.")
    A(f"2. **{len(multi)} targets keep a multi-chain prediction**: "
      f"{', '.join(r['target'] for r in multi)}.  Their interfaces are real and the reference "
      "scores 1.0000 against itself, but the arrangement is not reproduced: per-interface DockQ is "
      "0.005-0.434 in `dockq_interface_report.md`, above DockQ's 0.23 'acceptable' threshold only "
      "for 7W40_6 C-D (the receptor/G-protein core, 0.43).")
    A(f"3. **The chirality correction does not repair the quaternary structure.** The model chain "
      "count is identical for AF3, AT-AF3 and corrected AT-AF3 in every target and the pooled DockQ "
      "deltas stay within +/-0.005, but that pooling hides the per-interface picture: of the "
      "interfaces DockQ can score, 8U78_1 deteriorates on 9 of 12 (worst C-G -0.0317, 0/30 seeds "
      "improved) and 9BIC_1 on 6 of 19 (worst C-D -0.2677, F-G -0.1500) while other interfaces of "
      "the same model improve - see `dockq_interface_report.md` section 3 and "
      "`dockq_interface_paired.csv`.  The correction is a per-residue stereochemistry fix, not an "
      "assembly fix.")
    A("4. **Caveat on the racemic entries.** 8JH6_1 is annotated `dimeric`, but that dimer is the "
      "heterochiral D+L pair of a racemic crystallisation - the L copy is not a biological partner "
      "of the D target (8P6Q_2 is the same case, annotated monomeric), so the `1.0000` that DockQ "
      "now reports for those two files is the racemic pair, not a D-D assembly.  Read those "
      "references as single-chain references.")
    A("")
    A("Pooled DockQ of the predictions (from `gt_metrics_summary.csv`, mean over 30 seeds). "
      "**Read together with the caveat below the table:**")
    A("")
    A("| Target | Reference scope chains | Model chains | AF3 DockQ | Corrected DockQ |")
    A("|---|---|---|---|---|")
    for r in multi:
        vals = dq.get((r["tier"], r["target"]), {})

        def cell(v):
            return f"{v:.4f}" if v is not None and v == v else "n/a"

        A(f"| {r['target']} | {r['gt_chains']} | {r['model_chains']} | "
          f"{cell(vals.get('AF3'))} | {cell(vals.get('Corrected AT-AF3'))} |")
    A("")
    A("**DockQ caveat.**  These are the *pooled* values of the published suite.  "
      "DockQ's parser skips `HETATM` records by default and the canonical references write every "
      "D-residue that way, so each reference was scored as its L-subset only (7P5R_2 C/D 53$\\to$5 "
      "residues, 8GQP_1 A/C 62$\\to$2, 7QDI_1 31$\\to$2, 8U78_1 11$\\to$7, 9BIC_1 9$\\to$6, 7W40_6 F "
      "9$\\to$6) and the AT-AF3 predictions were hit by the same default.  The corrected evaluation "
      "keeps every D-residue, enumerates every reference interface and scores each one separately "
      "with an explicit `--mapping`: `scripts/dockq_interface_eval.py` $\\to$ "
      "`dockq_interface_report.md`, `dockq_interface_summary.csv`, `dockq_interface_paired.csv`.  "
      "Quote those, not this table.")
    A("")
    A("`n/a` (published table) marked the cases where the published suite could not locate a "
      "model-side interface - an artefact of the HETATM default, not an absent reference interface: "
      "8GQP_1 is scorable once the D-residues are retained (AF3 0.1192 / corrected 0.0930 on A-B).")
    A("")
    A("Sources (read-only):")
    A("")
    A("* `data/gt_reference/{all_d,local_d}/*.cif` - deposited assembly annotation.")
    A("* `configs/at_af_7.json`, `configs/target_chirality/*.json` - All-D entity/chain curation.")
    A("* `data/local_d/local_d_curated_v1_1/{06,22,23}_*.csv` - Local-D case, component "
      "and assembly-selection manifests.")
    A("* `results/gt_comparison_phenix_ccdfix/canonical_pdb/*_seed_0_GT.pdb` - chain inventory and "
      "inter-chain contacts of the scored reference.")
    A("* `reports_master42_n30_ccdfix/gt_metrics_summary.csv` - prediction chain counts and DockQ.")
    A("")
    with out_md.open("a") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Quaternary-structure inventory of the 16 targets")
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    ap.add_argument("--workdir", type=Path, default=None, help="temporary dir for the DockQ self-checks")
    ap.add_argument("--no-dockq", action="store_true", help="skip the DockQ reference self-check")
    args = ap.parse_args()
    rows = build_rows(args)
    out_md = args.out_dir / "quaternary_structure_table.md"
    out_csv = args.out_dir / "quaternary_structure_table.csv"
    yes, multi, n_dq = write_tables(rows, out_md, out_csv, not args.no_dockq)
    append_analysis(out_md, rows)
    print(f"wrote {out_md} and {out_csv.name} ({len(rows)} targets: {yes} with quaternary "
          f"structure, {multi} multi-chain predictions, {n_dq} scorable reference interfaces)")
    for r in rows:
        print(f"  {r['tier']:8s} {r['target']:9s} dep={r['deposited_assembly']:16s} "
              f"quat={r['quaternary']:3s} gt={r['gt_chains']:16s} model={r['model_chains']} "
              f"dockqGT={r['dockq_gt']}")


if __name__ == "__main__":
    main()
