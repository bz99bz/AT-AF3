"""Validate protein covalent bond geometry using Phenix/CCTBX restraints.

This is a read-only analysis.  It never edits coordinates, residue names, or
chirality and never launches AF3 or correction.  The analysis deliberately
does not infer bonds from inter-atomic distance thresholds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ALL_D_CASES = ("7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2")
LOCAL_D_CASES = ("7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1", "8U78_1", "8UM9_1", "9BIC_1")
SEEDS = (
    1373158607, 239081664, 53710185, 1592467582, 590620972,
    525901257, 479341424, 299655413, 1581559893, 220106708,
    1453201079, 1590571866, 1915941033, 1171165723, 186699714,
    1268073013, 906070221, 68252794, 63989048, 201209006,
    469521478, 499635469, 1085242217, 1292825379, 56985562,
    1205264596, 427000597, 1537640409, 1395616197, 1506083911,
)
STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
D_PROTEIN_RESIDUES = {
    "DAL", "DAR", "DAS", "DCY", "DGL", "DGN", "DHI", "DIL", "DLE", "DLY",
    "DME", "DPN", "DPR", "DSE", "DSG", "DTH", "DTR", "DTY", "DVA",
}
PROTEIN_RESIDUES = STANDARD_PROTEIN_RESIDUES | D_PROTEIN_RESIDUES


@dataclass(frozen=True)
class StructureRecord:
    dataset_type: str
    case: str
    seed: int
    path: Path
    stage: str = "corrected"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_structures(
    all_d_root: Path,
    local_d_root: Path,
    structure_filename: str = "corrected_all_errors.cif",
    stage: str = "corrected",
    selection: str = "exact",
) -> tuple[list[StructureRecord], list[dict]]:
    """Discover exactly one structure for every case/seed pair."""
    records: list[StructureRecord] = []
    issues: list[dict] = []
    expected = [("All-D", case, all_d_root) for case in ALL_D_CASES]
    expected += [("Local-D", case, local_d_root) for case in LOCAL_D_CASES]
    for dataset, case, root in expected:
        for seed in SEEDS:
            candidates = sorted(
                p for p in (root / case / f"seed_{seed}").rglob(structure_filename)
                if p.is_file()
            )
            if selection == "highest_ranking_score" and candidates:
                seed_root = root / case / f"seed_{seed}"
                ranking_files = sorted(seed_root.rglob("*_ranking_scores.csv"))
                if len(ranking_files) != 1:
                    candidates = []
                    issue = "missing_ranking" if not ranking_files else "duplicate_ranking"
                    issues.append({
                        "dataset_type": dataset, "case": case, "seed": seed,
                        "stage": stage, "issue": issue, "n_candidates": len(ranking_files),
                        "candidates": ";".join(str(p) for p in ranking_files),
                    })
                else:
                    with ranking_files[0].open(newline="", encoding="utf-8") as handle:
                        ranking_rows = list(csv.DictReader(handle))
                    scored = [(float(row["ranking_score"]), str(row["sample"]))
                              for row in ranking_rows]
                    if not scored:
                        candidates = []
                        issues.append({
                            "dataset_type": dataset, "case": case, "seed": seed,
                            "stage": stage, "issue": "empty_ranking", "n_candidates": 0,
                            "candidates": str(ranking_files[0]),
                        })
                    else:
                        _, selected_sample = max(scored)
                        candidates = [p for p in candidates if f"_sample-{selected_sample}_model.cif" in p.name]
            # The exact path normally yields zero or one.  Keep this explicit
            # so duplicate files cannot be selected implicitly.
            if len(candidates) != 1:
                issues.append({
                    "dataset_type": dataset,
                    "case": case,
                    "seed": seed,
                    "stage": stage,
                    "issue": "missing" if not candidates else "duplicate",
                    "n_candidates": len(candidates),
                    "candidates": ";".join(str(p) for p in candidates),
                })
                continue
            records.append(StructureRecord(dataset, case, seed, candidates[0], stage))
    return records, issues


def require_cctbx() -> None:
    missing = [name for name in ("iotbx", "mmtbx", "cctbx") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Phenix/CCTBX is required for restraint-backed bond geometry; "
            "missing imports: " + ", ".join(missing) + ". No RMSZ was calculated."
        )


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value: {value}")
    return value


def make_geometry_manager(path: Path):
    """Build the Phenix geometry manager and atom hierarchy for one CIF.

    The CCTBX API is intentionally imported only after the 480-structure
    completeness gate has passed.
    """
    from libtbx.utils import null_out
    from mmtbx.monomer_library import pdb_interpretation, server

    mon_lib_srv = server.server()
    ener_lib_srv = server.ener_lib()
    process_params = inspect.signature(pdb_interpretation.process).parameters
    kwargs = {"mon_lib_srv": mon_lib_srv, "file_name": str(path), "log": null_out()}
    interpretation_params = pdb_interpretation.master_params.extract()
    automatic = interpretation_params.automatic_linking
    automatic.link_all = False
    automatic.link_none = True
    automatic.link_metals = False
    automatic.link_residues = False
    automatic.link_amino_acid_rna_dna = False
    automatic.link_carbohydrates = False
    automatic.link_ligands = False
    automatic.link_small_molecules = False
    if "params" in process_params:
        kwargs["params"] = interpretation_params
    else:
        raise RuntimeError(
            "This CCTBX process API cannot accept pdb_interpretation params; "
            "automatic distance-based linking cannot be disabled safely."
        )
    if "make_restraints" in process_params:
        kwargs["make_restraints"] = True
    if "substitute_non_crystallographic_unit_cell_if_necessary" in process_params:
        kwargs["substitute_non_crystallographic_unit_cell_if_necessary"] = True
    if "ener_lib" in process_params:
        kwargs["ener_lib"] = ener_lib_srv
    elif "ener_lib_srv" in process_params:
        kwargs["ener_lib_srv"] = ener_lib_srv
    else:
        raise RuntimeError(
            "Unsupported CCTBX pdb_interpretation.process API: no "
            "ener_lib or ener_lib_srv parameter"
        )
    processed = pdb_interpretation.process(**kwargs)
    hierarchy = processed.all_chain_proxies.pdb_hierarchy
    sites_cart = hierarchy.atoms().extract_xyz()
    grm = processed.geometry_restraints_manager()
    if grm is None:
        xray_structure = getattr(processed, "_xray_structure", None)
        raise RuntimeError(
            f"CCTBX produced no geometry restraints manager for {path}; "
            "the parsed structure has "
            f"xray_structure={'present' if xray_structure is not None else 'missing'}. "
            "For AF3 prediction CIF files, verify support for "
            "substitute_non_crystallographic_unit_cell_if_necessary=True."
        )
    geometry = getattr(grm, "geometry", grm)
    if not hasattr(geometry, "get_all_bond_proxies"):
        raise RuntimeError(
            "Unsupported CCTBX geometry manager API: missing "
            "get_all_bond_proxies"
        )
    proxy_result = geometry.get_all_bond_proxies(sites_cart=sites_cart)
    if hasattr(proxy_result, "simple"):
        simple_proxies = proxy_result.simple
    elif isinstance(proxy_result, tuple) and len(proxy_result) >= 1:
        # Phenix 2.2/CCTBX returns (simple_proxies, asu_proxies).
        simple_proxies = proxy_result[0]
    else:
        raise RuntimeError(
            "Unsupported CCTBX get_all_bond_proxies return type: "
            f"{type(proxy_result).__name__}"
        )
    return hierarchy, simple_proxies


def _atom_info(atom):
    atom_group = atom.parent()
    residue_group = atom_group.parent()
    chain = residue_group.parent()
    return {
        "chain": str(chain.id),
        "resname": str(atom_group.resname).strip(),
        "resid": str(residue_group.resseq).strip(),
        "atom": str(atom.name).strip(),
        "altloc": str(atom_group.altloc).strip(),
        "element": str(atom.element).strip().upper(),
    }


def _bond_category(a: dict, b: dict) -> tuple[str, bool, bool]:
    same_residue = a["chain"] == b["chain"] and a["resid"] == b["resid"]
    if not same_residue:
        if a["atom"] == "C" and b["atom"] == "N" or a["atom"] == "N" and b["atom"] == "C":
            return "peptide", True, False
        return "disulfide_or_other_link", False, False
    backbone = {"N", "CA", "C", "O", "OXT"}
    if a["atom"] in backbone and b["atom"] in backbone:
        return "backbone", False, False
    if (a["atom"] in backbone) != (b["atom"] in backbone):
        return "backbone_sidechain", False, True
    return "sidechain", False, True


def _parent_resname(resname: str) -> str:
    mapping = {
        "DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS",
        "DGL": "GLU", "DGN": "GLN", "DHI": "HIS", "DIL": "ILE",
        "DLE": "LEU", "DLY": "LYS", "DME": "MET", "DPN": "PHE",
        "DPR": "PRO", "DSE": "SER", "DSG": "ASN", "DTH": "THR",
        "DTR": "TRP", "DTY": "TYR", "DVA": "VAL",
    }
    return mapping.get(resname, "")


def analyze_structure(record: StructureRecord) -> tuple[list[dict], dict, list[dict]]:
    cif_text = record.path.read_text(encoding="utf-8", errors="replace")
    if "_struct_conn." in cif_text or "_struct_conn_" in cif_text:
        raise RuntimeError(
            f"{record.path} declares _struct_conn links; explicit-link atom mapping "
            "must be resolved explicitly before inter-residue non-peptide proxies can be scored"
        )
    hierarchy, simple_proxies = make_geometry_manager(record.path)
    atoms = hierarchy.atoms()
    all_rows = []
    seen = set()
    qc = []
    unrecognized_residues = set()
    skipped_bonds = 0
    excluded_link_keys = set()
    sites = atoms.extract_xyz()
    for proxy in simple_proxies:
        i, j = map(int, proxy.i_seqs)
        if i == j or i >= len(atoms) or j >= len(atoms):
            qc.append({"case": record.case, "seed": record.seed, "stage": record.stage, "message": "invalid_bond_proxy_index"})
            continue
        ai, aj = atoms[i], atoms[j]
        a, b = _atom_info(ai), _atom_info(aj)
        unrecognized = [x for x in (a, b) if x["resname"] not in PROTEIN_RESIDUES]
        if unrecognized:
            skipped_bonds += 1
            for item in unrecognized:
                residue_key = (item["chain"], item["resid"], item["resname"])
                if residue_key not in unrecognized_residues:
                    unrecognized_residues.add(residue_key)
                    qc.append({
                        "case": record.case, "seed": record.seed, "stage": record.stage,
                        "structure_file": str(record.path),
                        "unrecognized_residue": item["resname"],
                        "chain": item["chain"], "resid": item["resid"],
                        "message": "non-protein or unresolved component excluded from bond metrics",
                    })
            continue
        if a["element"] in {"H", "D"} or b["element"] in {"H", "D"}:
            continue
        key = tuple(sorted(((a["chain"], a["resid"], a["atom"], a["altloc"]), (b["chain"], b["resid"], b["atom"], b["altloc"]))))
        if key in seen:
            qc.append({"case": record.case, "seed": record.seed, "stage": record.stage, "message": "duplicate_bond_proxy"})
            continue
        seen.add(key)
        d_obs = math.sqrt(sum((float(sites[i][q]) - float(sites[j][q])) ** 2 for q in range(3)))
        ideal = finite(proxy.distance_ideal)
        weight = finite(proxy.weight)
        if weight <= 0:
            qc.append({"case": record.case, "seed": record.seed, "stage": record.stage, "message": "invalid_sigma_weight"})
            continue
        sigma = math.sqrt(1.0 / weight)
        z = (d_obs - ideal) / sigma
        category, peptide, sidechain = _bond_category(a, b)
        if category == "peptide":
            try:
                adjacent = abs(int(a["resid"]) - int(b["resid"])) == 1
            except ValueError:
                adjacent = False
            if a["chain"] != b["chain"] or not adjacent:
                skipped_bonds += 1
                link_key = (a["chain"], a["resid"], a["atom"], b["chain"], b["resid"], b["atom"])
                if link_key not in excluded_link_keys:
                    excluded_link_keys.add(link_key)
                    qc.append({
                        "case": record.case, "seed": record.seed, "stage": record.stage,
                        "structure_file": str(record.path),
                        "message": "non-adjacent C-N proxy excluded from peptide metrics",
                    })
                continue
        elif category == "disulfide_or_other_link":
            # The evaluated AF3 CIF files contain no _struct_conn declarations.
            # CCTBX may still infer S-S links from geometry; these are not an
            # admissible topology source for this experiment.
            skipped_bonds += 1
            link_key = (a["chain"], a["resid"], a["atom"], b["chain"], b["resid"], b["atom"])
            if link_key not in excluded_link_keys:
                excluded_link_keys.add(link_key)
                qc.append({
                    "case": record.case, "seed": record.seed, "stage": record.stage,
                    "structure_file": str(record.path),
                    "message": "non-explicit inter-residue link excluded; no _struct_conn topology",
                })
            continue
        all_rows.append({
            "dataset_type": record.dataset_type, "case": record.case, "seed": record.seed,
            "stage": record.stage,
            "structure_file": str(record.path), "chain1": a["chain"], "resname1": a["resname"],
            "resid1": a["resid"], "atom1": a["atom"], "altloc1": a["altloc"],
            "chain2": b["chain"], "resname2": b["resname"], "resid2": b["resid"],
            "atom2": b["atom"], "altloc2": b["altloc"], "bond_category": category,
            "is_peptide_bond": peptide, "is_sidechain_bond": sidechain,
            "d_obs_A": d_obs, "d_ideal_A": ideal, "sigma_A": sigma,
            "delta_A": d_obs - ideal, "z_score": z, "abs_z": abs(z),
            "is_outlier_gt5": abs(z) > 5.0, "restraint_source": "Phenix/CCTBX bond proxy",
        })
    if not all_rows:
        raise RuntimeError(f"No heavy-atom bond proxies found for {record.path}")
    zs = [float(row["z_score"]) for row in all_rows]
    deltas = [float(row["delta_A"]) for row in all_rows]
    summary = {
        "dataset_type": record.dataset_type, "case": record.case, "seed": record.seed,
        "stage": record.stage,
        "structure_file": str(record.path), "n_bonds": len(all_rows),
        "n_bonds_rmsd": len(all_rows), "n_bonds_rmsz": len(all_rows),
        "bond_rmsd_A": math.sqrt(sum(d * d for d in deltas) / len(deltas)),
        "bond_rmsz": math.sqrt(sum(z * z for z in zs) / len(zs)),
        "n_outlier_gt5": sum(abs(z) > 5 for z in zs),
        "bond_outlier_rate_pct": 100.0 * sum(abs(z) > 5 for z in zs) / len(zs),
        "max_abs_z": max(abs(z) for z in zs), "mean_abs_delta_A": sum(abs(d) for d in deltas) / len(deltas),
        "n_unrecognized_residues": len(unrecognized_residues),
        "n_skipped_bonds": skipped_bonds, "status": "computed",
    }
    return all_rows, summary, qc


def analyze(args: argparse.Namespace) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records, issues = discover_structures(
        args.all_d_root,
        args.local_d_root,
        args.structure_filename,
        args.stage_label,
        args.selection,
    )
    write_csv(
        output / "missing_or_duplicate_structures.csv",
        issues,
        ["dataset_type", "case", "seed", "stage", "issue", "n_candidates", "candidates"],
    )
    expected = len(ALL_D_CASES + LOCAL_D_CASES) * len(SEEDS)
    if issues or len(records) != expected:
        (output / "README.md").write_text(
            "# Experiment A: Bond Geometry Validation\n\n"
            f"Structure completeness gate failed: expected {expected}, found {len(records)}.\n"
            "No bond geometry, RMSD, RMSZ, or outlier statistics were calculated.\n"
            "Missing/duplicate structures are listed in `missing_or_duplicate_structures.csv`.\n",
            encoding="utf-8",
        )
        print(f"STOP: expected {expected} structures, found {len(records)}; issues={len(issues)}")
        print(f"Wrote {output / 'missing_or_duplicate_structures.csv'}")
        return 2
    try:
        require_cctbx()
    except RuntimeError as exc:
        (output / "README.md").write_text(
            "# Experiment A: Bond Geometry Validation\n\n"
            "The 480-structure completeness gate passed, but the required "
            "Phenix/CCTBX restraint library is unavailable.\n\n"
            f"{exc}\n",
            encoding="utf-8",
        )
        print(f"STOP: {exc}")
        return 3
    bond_rows, structure_rows, qc_rows = [], [], []
    for record in records:
        bonds, summary, qc = analyze_structure(record)
        bond_rows.extend(bonds)
        structure_rows.append(summary)
        qc_rows.extend(qc)
    bond_fields = list(bond_rows[0])
    structure_fields = list(structure_rows[0])
    write_csv(output / "bond_geometry_all_bonds.csv", bond_rows, bond_fields)
    write_csv(output / "bond_geometry_per_structure.csv", structure_rows, structure_fields)
    write_csv(output / "bond_geometry_qc.csv", qc_rows,
              ["case", "seed", "structure_file", "unrecognized_residue", "chain", "resid", "message"])
    outliers = [row for row in bond_rows if row["is_outlier_gt5"]]
    outliers.sort(key=lambda row: float(row["abs_z"]), reverse=True)
    write_csv(output / "bond_length_outliers.csv", outliers, bond_fields)

    case_rows = []
    for (dataset, case, stage), group in _group_rows(structure_rows, ("dataset_type", "case", "stage")):
        case_rows.append(_aggregate_case(dataset, case, stage, group))
    write_csv(output / "bond_geometry_per_case.csv", case_rows, list(case_rows[0]))
    summary_rows = [_aggregate_dataset(dataset, [r for r in case_rows if r["dataset_type"] == dataset], args.stage_label)
                    for dataset in ("All-D", "Local-D")]
    summary_rows.append(_aggregate_dataset("Overall", case_rows, args.stage_label))
    write_csv(output / "bond_geometry_summary.csv", summary_rows, list(summary_rows[0]))
    _write_reference_table(output / "bond_reference_used.csv", bond_rows)
    _write_d_mapping(output / "d_residue_restraint_mapping.csv", bond_rows)
    _write_readme(output / "README.md", len(records), structure_rows)
    _write_markdown_table(output / "bond_geometry_table.md", summary_rows)
    print(f"Bond Geometry Validation complete: {len(structure_rows)} structures, {len(bond_rows)} bonds")
    return 0


def _group_rows(rows: list[dict], keys: tuple[str, ...]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups.items()


def _aggregate_case(dataset: str, case: str, stage: str, group: list[dict]) -> dict:
    def values(key): return [float(r[key]) for r in group]
    def mean(key):
        v = values(key)
        return sum(v) / len(v)
    def sd(key):
        v = values(key)
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"dataset_type": dataset, "case": case, "stage": stage, "n_seeds": len(group),
            "bond_rmsd_mean_A": mean("bond_rmsd_A"), "bond_rmsd_sd_A": sd("bond_rmsd_A"),
            "bond_rmsd_median_A": statistics.median(values("bond_rmsd_A")),
            "bond_rmsz_mean": mean("bond_rmsz"), "bond_rmsz_sd": sd("bond_rmsz"),
            "bond_rmsz_median": statistics.median(values("bond_rmsz")),
            "bond_outlier_rate_mean_pct": mean("bond_outlier_rate_pct"),
            "bond_outlier_rate_sd_pct": sd("bond_outlier_rate_pct"),
            "bond_outlier_rate_median_pct": statistics.median(values("bond_outlier_rate_pct"))}


def _aggregate_dataset(dataset: str, group: list[dict], stage: str) -> dict:
    def m(key): return sum(float(r[key]) for r in group) / len(group)
    def s(key):
        v = [float(r[key]) for r in group]; mean = sum(v) / len(v)
        return math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"dataset_type": dataset, "stage": stage, "n_cases": len(group), "n_structures": sum(int(r["n_seeds"]) for r in group),
            "bond_rmsd_mean_A": m("bond_rmsd_mean_A"), "bond_rmsd_sd_A": s("bond_rmsd_mean_A"),
            "bond_rmsz_mean": m("bond_rmsz_mean"), "bond_rmsz_sd": s("bond_rmsz_mean"),
            "bond_outlier_rate_mean_pct": m("bond_outlier_rate_mean_pct"),
            "bond_outlier_rate_sd_pct": s("bond_outlier_rate_mean_pct")}


def _write_reference_table(path: Path, rows: list[dict]) -> None:
    grouped = {}
    for row in rows:
        key = (row["resname1"], row["atom1"], row["resname2"], row["atom2"], row["bond_category"], row["d_ideal_A"], row["sigma_A"])
        grouped[key] = grouped.get(key, 0) + 1
    output = [{"residue_or_link_type": k[0] + "/" + k[2], "atom1": k[1], "atom2": k[3],
               "bond_context": k[4], "d_ideal_A": k[5], "sigma_A": k[6],
               "restraint_source": "Phenix/CCTBX bond proxy", "n_occurrences": n}
              for k, n in grouped.items()]
    write_csv(path, output, ["residue_or_link_type", "atom1", "atom2", "bond_context", "d_ideal_A", "sigma_A", "restraint_source", "n_occurrences"])


def _write_d_mapping(path: Path, rows: list[dict]) -> None:
    observed = set()
    for row in rows:
        for suffix in ("1", "2"):
            resname = str(row[f"resname{suffix}"])
            parent = _parent_resname(resname)
            if not parent:
                continue
            observed.add((resname, parent, row["case"], row["seed"], row[f"chain{suffix}"], row[f"resid{suffix}"]))
    output = [{"input_resname": item[0], "parent_resname": item[1], "case": item[2],
               "seed": item[3], "chain": item[4], "resid": item[5],
               "mapping_source": "Phenix/CCTBX direct D component; parent is used only for mapping"}
              for item in sorted(observed, key=lambda x: (x[2], int(x[3]), x[4], x[5], x[0]))]
    write_csv(path, output, ["input_resname", "parent_resname", "case", "seed", "chain", "resid", "mapping_source"])


def _write_readme(path: Path, n_structures: int, rows: list[dict]) -> None:
    path.write_text("# Experiment A: Bond Geometry Validation\n\n"
        f"Analyzed structures: {n_structures}. Hydrogen/deuterium atoms were excluded.\n\n"
        "Bonds are Phenix/CCTBX geometry-restraint bond proxies; no distance threshold was used to infer covalent bonds.\n\n"
        "Phenix automatic distance-based linking was disabled (`link_none=True`; metals, residues, "
        "carbohydrates, ligands, and small molecules all disabled). Standard monomer-library, "
        "polymer peptide, and explicit CIF-link restraints were retained.\n\n"
        "CCTBX-inferred inter-residue S-S/other links were excluded because the input prediction "
        "CIF files declared no `_struct_conn` topology. Only same-chain adjacent C--N peptide "
        "proxies were retained as inter-residue protein bonds.\n\n"
        "Bond RMSD is in Angstroms, Bond RMSZ is dimensionless, and outlier rate uses `|Z| > 5`.\n"
        "CCTBX bond-proxy weights were converted as sigma = sqrt(1 / weight), using the standard restraint weight convention.\n",
        encoding="utf-8")


def _write_markdown_table(path: Path, rows: list[dict]) -> None:
    lines = ["| Dataset | Cases | Structures | Bond RMSD (A) | Bond RMSZ | Outlier rate (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset_type']} | {r['n_cases']} | {r['n_structures']} | {r['bond_rmsd_mean_A']:.4f} +/- {r['bond_rmsd_sd_A']:.4f} | {r['bond_rmsz_mean']:.4f} +/- {r['bond_rmsz_sd']:.4f} | {r['bond_outlier_rate_mean_pct']:.4f} +/- {r['bond_outlier_rate_sd_pct']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-d-root", type=Path, default=Path("runs_master42_n30/at_af3_chirality_corrected"))
    parser.add_argument("--local-d-root", type=Path, default=Path("runs/local_d_curated_v1_1_post_cutoff/corrected"))
    parser.add_argument("--structure-filename", default="corrected_all_errors.cif")
    parser.add_argument(
        "--selection", choices=("exact", "highest_ranking_score"), default="exact",
        help="Select the unique model directly, or choose the highest-ranked AF3 sample per seed.",
    )
    parser.add_argument("--stage-label", default="corrected")
    parser.add_argument("--output-dir", type=Path, default=Path("results/bond_geometry"))
    return analyze(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
