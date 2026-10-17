"""Orchestration: manifest -> per-model mapping / preparation / detection -> deliverables.

For every (target, method, seed, sample) the GT is read and analysed **once** and reused; each
model is read, assigned its GT chains by one deterministic, one-to-one global correspondence
(``mapping.map_chains``: parent chemistry first, the declared target chain and the declared
reference chains of ``configs/target_chirality/<case>.json`` only break ties, chain ids only
break further ties), hydrogenated on a copy (Phenix ``reduce``) with its heavy atoms checked,
detected with the frozen detector, and compared against the same GT.  Nothing is minimised or
repaired.

Deliverables (in ``out_dir``):

* ``manifest.csv``                 one row per structure;
* ``residue_mapping.csv``          the verified residue correspondence to GT;
* ``chain_mapping.csv``            the chain correspondence behind it (``reference_chain`` /
  ``model_chain``, incl. unselected reference chains with their declared role and whether the
  declared scope requires them);
* ``preparation_audit.csv``        hydrogenation + heavy-atom + altloc record per structure;
* ``interactions.csv``             the common interaction records for every method (incl. GT);
* ``interaction_comparison_per_model.csv``  retained / distorted / lost / extra per model;
* ``interaction_comparison_summary.csv``    per model per class counts, geometry-aware recall/
  precision/F1, strict identity metrics and an explicit ``reference_status`` (a class present in
  GT but entirely absent from a model - e.g. protein-metal or water-mediated - gets a
  ``reference_only_model_absent`` row);
* ``chemistry_audit.csv``          residue-name -> chemistry resolution record;
* ``tool_inventory.csv``           Phenix / reduce version and probe results;
* ``audit.json`` and ``README.md`` hold the run's frozen protocol and provenance.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import chem as C
from . import interactions as I
from . import manifest as MF
from . import mapping as M
from . import prepare as PREP
from . import protocol as P
from .structure_io import read_structure

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_pdb"
DEFAULT_CHAIN_MAP = ROOT / "results/gt_comparison_phenix_ccdfix/canonical_chain_id_mapping.json"
DEFAULT_CHIRALITY = ROOT / "configs/target_chirality"
DEFAULT_CCD_CACHE = ROOT / "data/ccd_components_cache.json"
DEFAULT_OUT = ROOT / "results/phenix_interaction"
DEFAULT_REDUCE = os.environ.get("PHENIX_REDUCE", "phenix.reduce")

MANIFEST_COLS = ["target", "task", "method", "seed", "sample", "structure_path",
                 "reference_path", "selected_chains", "dataset", "condition_suffix",
                 "source_path", "sha256", "exists", "expected_cofactors", "notes"]
#: one-line statement of the correspondence rule, published in ``README.md`` and ``audit.json``
CHAIN_RULE = (
    "Every model chain is paired with at most one reference chain by the global one-to-one "
    "assignment that maximises the total parent-chemistry agreement, except the declared target "
    "chain, which claims the model chain it reproduces best before that assignment runs.  The raw "
    "aligned-position objective overrode the declared reference copy; declared-target assignment "
    "is now enforced first.")
RESIDUE_COLS = ["target", "dataset", "seed", "sample", "method", "reference_chain", "model_chain",
                "gt_residue", "gt_resseq", "gt_icode", "gt_resname", "gt_chem_class", "gt_is_D",
                "model_residue", "model_resseq", "model_icode", "model_resname",
                "model_chem_class", "model_is_D", "mapped", "mapping_type", "gt_supported",
                "model_supported", "gt_heavy_atoms", "model_heavy_atoms", "heavy_atoms_matched",
                "in_domain", "is_target_chain", "notes"]
CHAIN_COLS = ["target", "dataset", "seed", "sample", "method", "reference_chain", "model_chain",
              "matched", "assignment_basis", "reference_role", "required", "selection",
              "identical_residues", "aligned_residues", "identity", "model_coverage",
              "d_fraction_delta"]
PREP_COLS = ["target", "dataset", "seed", "sample", "method", "input_path", "heavy_input_pdb",
             "altloc_policy", "altloc_residues", "altloc_atoms_dropped", "reduce_status",
             "reduce_version", "protonation_policy", "altered_resnames", "heavy_atoms_input",
             "heavy_atoms_hydrogenated", "heavy_atoms_matched", "heavy_atoms_missing",
             "heavy_atoms_added", "max_heavy_atom_shift", "hydrogens_added",
             "heavy_atoms_unchanged", "usable", "reason"]
INTERACTION_COLS = ["interaction_id", "target", "dataset", "seed", "sample", "method", "scope",
                    "interaction_class", "chain_a", "res_a", "resname_a", "chem_class_a",
                    "is_d_a", "atom_a", "element_a", "chain_b", "res_b", "resname_b",
                    "chem_class_b", "is_d_b", "atom_b", "element_b", "distance", "angle",
                    "angle_partner", "hydrogen_source", "sequence_separation", "subset",
                    "cutoff_name", "cutoff_value", "geometry_band", "evaluation_status",
                    "reference_id", "delta_distance", "delta_angle", "notes"]
COMPARISON_COLS = ["target", "dataset", "seed", "sample", "method", "interaction_class",
                   "scope", "subset", "status", "residue_pair", "atom_pair", "gt_distance",
                   "model_distance", "delta_distance", "gt_angle", "model_angle", "delta_angle",
                   "provenance"]
SUMMARY_COLS = ["target", "dataset", "seed", "sample", "method", "interaction_class", "scope",
                "n_reference", "n_model", "retained", "retained_but_distorted", "lost", "extra",
                "outside_evaluable_domain", "recall", "precision", "f1",
                "recall_identity", "precision_identity", "f1_identity", "reference_status"]


@dataclass
class Settings:
    canonical_dir: Path = DEFAULT_CANONICAL
    chain_map_json: Path = DEFAULT_CHAIN_MAP
    chirality_dir: Path = DEFAULT_CHIRALITY
    ccd_cache: Path = DEFAULT_CCD_CACHE
    out_dir: Path = DEFAULT_OUT
    workdir: Path = ROOT / "results/phenix_interaction/work"
    reduce_tool: str = str(DEFAULT_REDUCE)
    cases: list = field(default_factory=list)
    max_seeds: int = 0
    altloc_policy: str = "best"
    use_reduce: bool = True


def write_csv(path: Path, rows: list, columns: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def cofactor_summary(structure) -> str:
    """Describe the non-polymer content a reader needs (waters, metals, ligands)."""
    waters = sum(1 for _ in structure.water_residues())
    metals = {}
    others = {}
    for res in structure.residues():
        if res.is_water or res.is_polymer:
            continue
        for a in res.heavy_atoms():
            if a.element in P.METAL_ELEMENTS:
                metals[a.element] = metals.get(a.element, 0) + 1
        others[res.resname] = others.get(res.resname, 0) + 1
    parts = []
    if metals:
        parts.append("metals=" + ",".join(f"{k}:{v}" for k, v in sorted(metals.items())))
    if others:
        parts.append("ligands=" + ",".join(f"{k}:{v}" for k, v in sorted(others.items())))
    parts.append(f"waters={waters}")
    return "; ".join(parts)


def _declared_chains(case: str, chirality_dir: Path) -> tuple:
    """``(target_chain, declared_chains)`` declared by ``configs/target_chirality/<case>.json``.

    The keys of ``chains`` are the GT reference chains the case declares (the same declaration
    ``build_target_scope.py`` publishes as ``reference_chains``); the target chain is the first
    declared chain designed as the D copy, falling back to the first declared chain.  A case
    without a chirality config (the Local-D cases) declares no chain explicitly: it returns
    ``("", ())`` and the correspondence is then decided by the parent chemistry and the chain
    ids, which reproduces the declared identity of requested and reference chains.
    """
    cfg = Path(chirality_dir) / f"{case}.json"
    if not cfg.exists():
        return "", ()
    try:
        data = json.loads(cfg.read_text())
        chains = data.get("chains") or {}
        declared = [str(chain) for chain in chains]
    except Exception:
        return "", ()
    if not declared:
        return "", ()
    target = ""
    for chain in declared:
        spec = chains.get(chain) or {}
        if str(spec.get("default_target_chirality", "")).upper() == "D":
            target = chain
            break
    return (target or declared[0]), tuple(declared)


def _meta(row) -> dict:
    return {"target": row["target"], "dataset": row["dataset"], "seed": row["seed"],
            "sample": row["sample"], "method": row["method"]}


def _prepare_struct(struct, row, settings, resolver, tag):
    """Hydrogenate one structure (or report why not); return ``(audit, placed, heavy_pdb)``."""
    if settings.use_reduce and settings.reduce_tool:
        _reduced, audit, placed = PREP.hydrogenate(
            struct, settings.reduce_tool, Path(settings.workdir), tag, resolver)
    else:
        audit = {"reduce_tool": P.NA, "reduce_version": P.NA,
                 "protonation_policy": P.PROTONATION_POLICY, "altered_resnames": "",
                 "heavy_atoms_input": struct.heavy_atom_count(), "heavy_atoms_hydrogenated": 0,
                 "heavy_atoms_matched": 0, "heavy_atoms_missing": 0, "heavy_atoms_added": 0,
                 "max_heavy_atom_shift": P.NA, "hydrogens_added": 0,
                 "reduce_status": "disabled", "heavy_atoms_unchanged": "yes", "usable": False,
                 "reason": "reduce disabled"}
        placed = {}
    heavy_pdb = str(Path(settings.workdir) / f"{tag}_heavy.pdb")
    return audit, placed, heavy_pdb


def _prep_row(row, audit, read_audit, heavy_pdb) -> dict:
    out = {"target": row["target"], "dataset": row["dataset"], "seed": row["seed"],
           "sample": row["sample"], "method": row["method"],
           "input_path": row["structure_path"], "heavy_input_pdb": heavy_pdb,
           "altloc_policy": read_audit.get("altloc_policy", P.NA),
           "altloc_residues": read_audit.get("alternate_locator_residues", P.NA),
           "altloc_atoms_dropped": read_audit.get("alternate_locator_atoms_dropped", P.NA)}
    out.update(audit)
    if isinstance(out.get("usable"), bool):
        out["usable"] = "yes" if out["usable"] else "no"
    return out


def _targets(manifest_rows, cases, max_seeds):
    targets = {}
    for r in manifest_rows:
        if cases and r["target"] not in cases:
            continue
        targets.setdefault((r["dataset"], r["target"]), []).append(r)
    if max_seeds:
        for rows in targets.values():
            keep_gt = [r for r in rows if r["method"] == P.METHOD_GT]
            bym = {}
            for r in rows:
                if r["method"] != P.METHOD_GT:
                    bym.setdefault(r["method"], []).append(r)
            trimmed = list(keep_gt)
            for m in P.MODEL_METHODS:
                trimmed += sorted(bym.get(m, []), key=lambda x: int(x["seed"]))[:max_seeds]
            rows[:] = trimmed
    return targets


def run(settings, progress=lambda message: None):
    resolver = C.ChemResolver(ccd_cache=settings.ccd_cache)
    manifest_rows = MF.build_manifest(settings.canonical_dir, settings.chain_map_json,
                                      settings.chirality_dir)
    targets = _targets(manifest_rows, set(settings.cases or []), settings.max_seeds)

    all_records = []
    residue_rows = []
    chain_rows = []
    prep_rows = []
    comparison_rows = []

    for (dataset, case), rows in sorted(targets.items()):
        gt_rows = [r for r in rows if r["method"] == P.METHOD_GT]
        if not gt_rows:
            progress(f"[{case}] no GT row; skipped")
            continue
        gt_row = gt_rows[0]
        gt_struct, gt_read = read_structure(gt_row["structure_path"], name=f"{case}_GT",
                                            resolver=resolver, altloc_policy=settings.altloc_policy)
        cofactors = cofactor_summary(gt_struct)
        for r in rows:
            r["expected_cofactors"] = cofactors
        target_chain, declared_chains = _declared_chains(case, settings.chirality_dir)
        gt_meta = _meta(gt_row)
        gt_audit, gt_placed, gt_heavy = _prepare_struct(gt_struct, gt_row, settings, resolver,
                                                        f"{case}_GT_seed{gt_row['seed']}")
        prep_rows.append(_prep_row(gt_row, gt_audit, gt_read, gt_heavy))
        gt_recs = I.detect_all(gt_struct, gt_placed, gt_meta, resolver)
        for rec in gt_recs:
            rec.evaluation_status = P.EVAL_REFERENCE
        all_records.extend(gt_recs)
        progress(f"[{case}] ({dataset}) GT: {len(gt_recs)} records; {cofactors}")

        model_rows = [r for r in rows if r["method"] != P.METHOD_GT]
        for row in sorted(model_rows, key=lambda x: (P.MODEL_METHODS.index(x["method"]),
                                                     int(x["seed"]))):
            tag = f"{case}_{row['method'].replace(' ', '')}_{row['seed']}"
            model_struct, read_audit = read_structure(row["structure_path"], name=tag,
                                                      resolver=resolver,
                                                      altloc_policy=settings.altloc_policy)
            cmaps = M.map_chains(model_struct, gt_struct, target_chain, declared_chains)
            for cr in M.chain_mapping_rows(gt_struct, cmaps, target_chain, declared_chains):
                cr.update({"target": case, "dataset": dataset, "seed": row["seed"],
                           "sample": row["sample"], "method": row["method"]})
                chain_rows.append(cr)
            rrows, forward, _reverse = M.residue_mapping(model_struct, gt_struct, cmaps,
                                                         target_chain)
            for rr in rrows:
                rr.update({"target": case, "dataset": dataset, "seed": row["seed"],
                           "sample": row["sample"], "method": row["method"]})
            residue_rows.extend(rrows)
            prep_audit, placed, heavy_pdb = _prepare_struct(model_struct, row, settings, resolver,
                                                            tag)
            prep_rows.append(_prep_row(row, prep_audit, read_audit, heavy_pdb))
            meta = _meta(row)
            recs = I.detect_all(model_struct, placed, meta, resolver)
            all_records.extend(recs)
            comparison_rows.extend(I.compare_to_reference(gt_recs, recs, forward, meta))
            mapped = sum(1 for x in rrows if x["mapped"] == "yes")
            progress(f"[{case}] {row['method']} seed={row['seed']} sample={row['sample']}: "
                     f"{mapped} mapped, {len(recs)} records, reduce={prep_audit['reduce_status']}; "
                     f"chains: "
                     f"{M.correspondence_summary(gt_struct, cmaps, target_chain, declared_chains)}")

    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    interaction_rows = [rec.as_row(i) for i, rec in enumerate(all_records)]
    summary_rows = _summarise(comparison_rows, _gt_reference_counts(all_records))
    selected_rows = [r for rows in targets.values() for r in rows]
    selected_rows.sort(key=lambda r: (r["dataset"], r["target"], r["method"], int(r["seed"])))
    counts = {
        "manifest.csv": write_csv(out / "manifest.csv", selected_rows, MANIFEST_COLS),
        "residue_mapping.csv": write_csv(out / "residue_mapping.csv", residue_rows, RESIDUE_COLS),
        "chain_mapping.csv": write_csv(out / "chain_mapping.csv", chain_rows, CHAIN_COLS),
        "preparation_audit.csv": write_csv(out / "preparation_audit.csv", prep_rows, PREP_COLS),
        "interactions.csv": write_csv(out / "interactions.csv", interaction_rows,
                                      INTERACTION_COLS),
        "interaction_comparison_per_model.csv":
            write_csv(out / "interaction_comparison_per_model.csv", comparison_rows,
                      COMPARISON_COLS),
        "interaction_comparison_summary.csv":
            write_csv(out / "interaction_comparison_summary.csv", summary_rows, SUMMARY_COLS),
        "chemistry_audit.csv": write_csv(out / "chemistry_audit.csv", resolver.audit_rows(),
                                         ["resname", "chem_class", "parent", "is_D", "is_polymer",
                                          "is_water", "supported", "source", "note"]),
        "tool_inventory.csv": write_csv(out / "tool_inventory.csv", _tool_inventory(settings),
                                        ["tool", "path", "version", "role", "status"]),
    }
    _write_audit(out, settings, counts, _unselected_reference_summary(chain_rows))
    return {"deliverables": counts, "targets": sorted({k[1] for k in targets}),
            "n_interactions": len(interaction_rows), "n_comparison": len(comparison_rows)}


def _ratio(num: int, den: int):
    return round(num / den, 4) if den else P.NA


def _f1(recall, precision):
    if isinstance(recall, float) and isinstance(precision, float) and (recall + precision):
        return round(2 * recall * precision / (recall + precision), 4)
    return P.NA


def _gt_reference_counts(records) -> dict:
    """``{(target, dataset): {(interaction_class, scope): n}}`` over the GT records.

    This counts **every** reference interaction, including the ones whose partner is a
    non-polymer residue (metal ion, water, cofactor) and which therefore never enter the
    polymer-only comparison domain.  ``_summarise`` uses it to emit an explicit row for a
    class/scope that exists in GT but is entirely absent from the model.
    """
    counts: dict = {}
    for rec in records:
        if getattr(rec, "method", "") != P.METHOD_GT:
            continue
        by_cls = counts.setdefault((rec.target, rec.dataset), {})
        key = (rec.interaction_class, rec.scope)
        by_cls[key] = by_cls.get(key, 0) + 1
    return counts


def _summarise(comparison_rows, gt_reference_counts: dict | None = None) -> list:
    """Aggregate the per-model comparison into per (model, class, scope) metrics.

    Both a geometry-aware and a strict identity metric are reported:

    * ``retained`` counts a match whose geometry is preserved within the frozen band; a
      ``retained_but_distorted`` match keeps the interaction *identity* (same class, scope and
      residue/atom pair) but moves past the distance/angle tolerance.  ``recall``/``precision``/
      ``f1`` use ``retained + retained_but_distorted`` (identity-preserving, geometry-aware).
    * ``recall_identity``/``precision_identity``/``f1_identity`` use ``retained`` only, i.e. they
      separate pure identity preservation from geometric distortion.

    Classes present in GT but never produced by the model (metal coordination, water-mediated
    contacts) get an explicit ``reference_only_model_absent`` row, so an empty model class is
    distinguishable from "not evaluated".
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"retained": 0, "retained_but_distorted": 0, "lost": 0,
                               "extra": 0, "outside_evaluable_domain": 0})
    for r in comparison_rows:
        key = (r["target"], r["dataset"], r["seed"], r["sample"], r["method"],
               r["interaction_class"], r["scope"])
        if r["status"] in agg[key]:
            agg[key][r["status"]] += 1
    synthetic = set()
    if gt_reference_counts:
        present = defaultdict(set)
        for r in comparison_rows:
            present[(r["target"], r["dataset"], r["seed"], r["sample"], r["method"])].add(
                (r["interaction_class"], r["scope"]))
        for mkey, cls_keys in present.items():
            for (cls, scope), n in gt_reference_counts.get((mkey[0], mkey[1]), {}).items():
                if (cls, scope) in cls_keys:
                    continue
                key = (mkey[0], mkey[1], mkey[2], mkey[3], mkey[4], cls, scope)
                agg[key]["lost"] += n        # every reference record is unreproduced by the model
                synthetic.add(key)
    out = []
    for key, a in sorted(agg.items()):
        matched = a["retained"] + a["retained_but_distorted"]
        n_ref = matched + a["lost"]
        n_mod = matched + a["extra"]
        recall, precision = _ratio(matched, n_ref), _ratio(matched, n_mod)
        recall_i, precision_i = _ratio(a["retained"], n_ref), _ratio(a["retained"], n_mod)
        if key in synthetic:
            status = P.REFERENCE_MODEL_ABSENT
        elif n_ref:
            status = P.REFERENCE_COMPARED
        else:
            status = P.REFERENCE_NONE
        out.append({"target": key[0], "dataset": key[1], "seed": key[2], "sample": key[3],
                    "method": key[4], "interaction_class": key[5], "scope": key[6],
                    "n_reference": n_ref, "n_model": n_mod, "retained": a["retained"],
                    "retained_but_distorted": a["retained_but_distorted"], "lost": a["lost"],
                    "extra": a["extra"], "outside_evaluable_domain": a["outside_evaluable_domain"],
                    "recall": recall, "precision": precision, "f1": _f1(recall, precision),
                    "recall_identity": recall_i, "precision_identity": precision_i,
                    "f1_identity": _f1(recall_i, precision_i), "reference_status": status})
    return out


def _phenix_version(reduce_tool: str) -> str:
    try:
        enter = Path(reduce_tool).resolve().parents[1]
        vcmd = enter / "phenix_bin" / "phenix.version"
        if vcmd.exists():
            import subprocess
            out = subprocess.run([str(vcmd)], capture_output=True, text=True, timeout=60)
            for line in ((out.stdout or "") + (out.stderr or "")).splitlines():
                if "Release tag" in line or "Version:" in line:
                    return line.strip()
    except Exception:
        pass
    return "unavailable"


def _tool_inventory(settings) -> list:
    import shutil
    reduce_tool = settings.reduce_tool
    rows = [
        {"tool": "phenix.reduce", "path": reduce_tool,
         "version": PREP.reduce_version(reduce_tool),
         "role": "hydrogen placement for hbond angles",
         "status": "available" if reduce_tool and Path(reduce_tool).exists() else "missing"},
        {"tool": "phenix (iotbx / cctbx)", "path": str(Path(reduce_tool).resolve().parents[1])
         if reduce_tool else P.NA, "version": _phenix_version(reduce_tool),
         "role": "PDB/mmCIF parsing, residue & atom correspondence, geometry", "status": "available"},
    ]
    for name in ("phenix.probe", "phenix.ramalyze", "phenix.clashscore", "phenix.mtz2map"):
        path = shutil.which(name)
        rows.append({"tool": name, "path": path or P.NA, "version": "", "role": "probe",
                     "status": "available" if path else "missing"})
    return rows


def _unselected_reference_summary(chain_rows) -> dict:
    """Declared role of every reference chain the run left unselected (``audit.json``).

    A reference chain that no model chain claimed is grouped per (target, method, reference chain)
    with the declared role it was judged by.  ``required=yes`` means the declared scope (or the
    undeclared whole-complex scope of the Local-D cases) does contain it, so it is a real gap;
    ``required=no`` is a copy outside the declared scope - 8BFD_2 chain A, the L enantiomer the
    entry also deposits - which must not be counted as a missing required chain.
    """
    groups = {}
    for row in chain_rows:
        if row["reference_chain"] == P.NA or row["matched"] == "yes":
            continue
        key = (row["target"], row["method"], row["reference_chain"])
        entry = groups.setdefault(key, {"dataset": row["dataset"],
                                        "reference_role": row["reference_role"],
                                        "required": row["required"],
                                        "selection": row["selection"], "n_seed_rows": 0})
        entry["n_seed_rows"] += 1
    by_role = {}
    for entry in groups.values():
        by_role[entry["reference_role"]] = by_role.get(entry["reference_role"], 0) + 1
    return {"n_group_keys": len(groups), "by_role": by_role,
            "n_required": sum(1 for e in groups.values() if e["required"] == "yes"),
            "n_not_required": sum(1 for e in groups.values() if e["required"] == "no"),
            "groups": [dict({"target": key[0], "method": key[1], "reference_chain": key[2]}, **entry)
                       for key, entry in sorted(groups.items())]}


def _write_audit(out: Path, settings, counts, unselected=None) -> None:
    audit = {
        "protocol_id": P.PROTOCOL_ID,
        "protocol_choices": P.PROTOCOL_CHOICES,
        "provenance_notes": P.PROVENANCE_NOTES,
        "chain_correspondence_rule": CHAIN_RULE,
        "unselected_reference_chains": unselected or {},
        "settings": {
            "canonical_dir": str(settings.canonical_dir),
            "chain_map_json": str(settings.chain_map_json),
            "chirality_dir": str(settings.chirality_dir),
            "ccd_cache": str(settings.ccd_cache),
            "reduce_tool": settings.reduce_tool,
            "altloc_policy": settings.altloc_policy,
            "max_seeds": settings.max_seeds,
            "cases": list(settings.cases or []),
            "use_reduce": settings.use_reduce,
        },
        "tools": _tool_inventory(settings),
        "deliverables": counts,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    readme = [
        "# Phenix-based interaction preservation analysis",
        "",
        "Compares GT, standard AF3, uncorrected AT-AF3 and corrected AT-AF3 for disulfides, salt",
        "bridges, metal coordination, hydrogen bonds and hydrophobic contacts, with the atom-level",
        "geometry saved as evidence.  Structures are parsed with iotbx/cctbx; hydrogens are placed",
        "on a copy with Phenix `reduce` (`-BUILD -NOFLIP`) and every heavy atom is checked to be",
        "unchanged.  Nothing is minimised, rebuilt, re-flipped or repaired.",
        "",
        "## Deliverables",
        "",
    ]
    for name in sorted(counts):
        readme.append(f"* `{name}` ({counts[name]} rows)")
    readme += [
        "",
        "## Chain correspondence",
        "",
        CHAIN_RULE,
        "",
        "`chain_mapping.csv` names the two sides explicitly (`reference_chain`, `model_chain`) and "
        "carries the declared role of every reference chain it reports (`reference_role`, "
        "`required`, `selection`).  A reference chain no model chain claimed is therefore readable "
        "as \"not part of the task\" instead of \"missing\": 8BFD_2 chain A, the L enantiomer the "
        "entry also deposits next to the declared D copy B, is `outside_declared_scope` / "
        "`required=no`, and the geometry run classifies its references as "
        "`outside_requested_scope`, never as `missing_required_chain`.",
    ]
    if unselected:
        readme += ["", "Unselected reference chains of this run "
                       f"({unselected.get('n_required', 0)} required, "
                       f"{unselected.get('n_not_required', 0)} outside the declared scope):", ""]
        for group in unselected.get("groups", ()):
            readme.append(f"* {group['target']} {group['method']}: reference chain "
                          f"`{group['reference_chain']}` - `{group['reference_role']}`, "
                          f"required={group['required']}, {group['selection']} "
                          f"({group['n_seed_rows']} model file(s))")
    readme += ["", f"Protocol id: `{P.PROTOCOL_ID}`  ", f"Provenance: {P.PROVENANCE_NOTES}"]
    (out / "README.md").write_text("\n".join(readme) + "\n")


