"""Local chirality mirror correction for binary error-code rows."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .chirality_geometry import (
    CHIRAL_ATOMS,
    FIXED_BACKBONE_ATOMS,
    atom_coord,
    classify_residue_chirality,
    geometry_quality,
    iter_residues,
    reflect_point_across_plane,
    set_atom_coord,
)

VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
_PARENT = {"DAL":"ALA", "DAR":"ARG", "DAS":"ASP", "DCY":"CYS", "DGL":"GLU", "DGN":"GLY", "DHI":"HIS", "DIL":"ILE", "DLE":"LEU", "DLY":"LYS", "DME":"MET", "DPN":"PRO", "DPR":"PRO", "DSE":"SER", "DSG":"GLY", "DTH":"THR", "DTR":"TRP", "DTY":"TYR", "DVA":"VAL"}

_SIDECHAIN_BONDS = {
    "ALA": [("CA", "CB")], "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "CYS": [("CA", "CB"), ("CB", "SG")], "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CG1", "CD1"), ("CG1", "CG2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2")],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
}


def _bond_pairs(residue_atoms: list[dict]) -> list[tuple[dict, dict]]:
    resname = str(residue_atoms[0].get("resname", "")).upper() if residue_atoms else ""
    pairs = _SIDECHAIN_BONDS.get(_PARENT.get(resname, resname), [])
    by_name = {a.get("name", ""): a for a in residue_atoms}
    return [(by_name[x], by_name[y]) for x, y in pairs if x in by_name and y in by_name]


def _heavy(atom: dict) -> bool:
    return str(atom.get("element", "")).upper() not in {"H", ""} and not str(atom.get("name", "")).upper().startswith("H")


def _overlap(a: dict, b: dict) -> float:
    ra = VDW_RADII.get(str(a.get("element", "")).upper(), 1.70)
    rb = VDW_RADII.get(str(b.get("element", "")).upper(), 1.70)
    return ra + rb - float(np.linalg.norm(atom_coord(a) - atom_coord(b)))


def _sanity_check(before: list[dict], after: list[dict], context: list[dict], max_displacement_A: float, max_bond_change_A: float, severe_overlap_A: float) -> dict:
    before_by = {a.get("name", ""): a for a in before}; after_by = {a.get("name", ""): a for a in after}
    moves = [(float(np.linalg.norm(atom_coord(after_by[n]) - atom_coord(before_by[n]))), n) for n in after_by if n in before_by and n not in FIXED_BACKBONE_ATOMS]
    bond_rows = []
    for x, y in _bond_pairs(before):
        if x.get("name") not in after_by or y.get("name") not in after_by: continue
        d0=float(np.linalg.norm(atom_coord(x)-atom_coord(y))); d1=float(np.linalg.norm(atom_coord(after_by[x.get("name")])-atom_coord(after_by[y.get("name")]))); bond_rows.append({"atom1":x.get("name"),"atom2":y.get("name"),"before_A":d0,"after_A":d1,"delta_A":d1-d0})
    context_other=[a for a in context if not (a.get("chain")==before[0].get("chain") and int(a.get("resid",-1))==int(before[0].get("resid",-2)))]
    clashes=[]
    for n, aa in after_by.items():
        if n in FIXED_BACKBONE_ATOMS or not _heavy(aa): continue
        old=before_by.get(n)
        for other in context_other:
            if not _heavy(other): continue
            ov1=_overlap(aa, other)
            if ov1 < severe_overlap_A: continue
            ov0=_overlap(old, other) if old is not None else -math.inf
            if ov0 < severe_overlap_A:
                clashes.append({"mirrored_atom":n,"clash_atom":f"{other.get('chain')}:{other.get('resid')}:{other.get('name')}","distance_before_A":float(np.linalg.norm(atom_coord(old)-atom_coord(other))) if old is not None else math.nan,"distance_after_A":float(np.linalg.norm(atom_coord(aa)-atom_coord(other))),"displacement_A":float(np.linalg.norm(atom_coord(aa)-atom_coord(old))) if old is not None else math.nan,"overlap_after_A":ov1})
    max_move=max((x[0] for x in moves),default=0.0); max_bond=max((abs(x["delta_A"]) for x in bond_rows),default=0.0)
    return {"max_atom_displacement":max_move,"max_bond_delta_A":max_bond,"bond_rows":bond_rows,"new_severe_clashes":clashes,"accepted":max_move<=max_displacement_A and max_bond<=max_bond_change_A and not clashes,"reject_reasons":[x for x,ok in (("atom_displacement",max_move<=max_displacement_A),("bond_geometry",max_bond<=max_bond_change_A),("new_severe_vdw_clash",not clashes)) if not ok]}


def _is_error_code_one(value: object) -> bool:
    """Accept in-memory integer rows and rows reloaded from error-code CSV files."""
    return value == 1 or str(value).strip() == "1"


def _sidechain_changed(before_atoms: list[dict], after_atoms: list[dict]) -> bool:
    """Infer a local sidechain transform for standalone before/after evaluation."""
    after_map = {atom.get("name", ""): atom for atom in after_atoms}
    for atom in before_atoms:
        if atom.get("name") in FIXED_BACKBONE_ATOMS:
            continue
        other = after_map.get(atom.get("name", ""))
        if other is not None and not np.allclose(atom_coord(atom), atom_coord(other), atol=1e-6):
            return True
    return False


def correction_keys_from_error_package(package: dict, correction_mode: str = "all", max_corrections: int | None = None) -> set[tuple[str, int]]:
    if correction_mode != "all":
        raise ValueError("Only correction-mode 'all' is supported for formal runs")
    keys: list[tuple[str, int]] = []
    for chain, residues in package.get("residues", {}).items():
        for row in residues:
            if _is_error_code_one(row.get("error_code")) and row.get("correction_eligible") is True:
                keys.append((chain, int(row["residue_index"])))
    if max_corrections is not None:
        keys = keys[: int(max_corrections)]
    return set(keys)


def mirror_residue_sidechain(
    residue_atoms: list[dict],
    context_atoms: list[dict] | None = None,
    max_displacement_A: float = 5.0,
    max_bond_change_A: float = 0.2,
    severe_overlap_A: float = 0.4,
) -> tuple[list[dict], dict]:
    atom_map = {a.get("name", ""): a for a in residue_atoms}
    missing = [name for name in CHIRAL_ATOMS if name not in atom_map]
    if missing:
        return residue_atoms, {
            "correction_applied": False,
            "n_atoms_mirrored": 0,
            "reason": "missing_atoms:" + ",".join(missing),
            "mean_atom_displacement": 0.0,
            "max_atom_displacement": 0.0,
        }
    n = atom_coord(atom_map["N"])
    ca = atom_coord(atom_map["CA"])
    c = atom_coord(atom_map["C"])
    normal = np.cross(n - ca, c - ca)
    mirrored = []
    displacements = []
    for atom in residue_atoms:
        new_atom = copy.deepcopy(atom)
        if atom.get("name") not in FIXED_BACKBONE_ATOMS:
            before = atom_coord(atom)
            after = reflect_point_across_plane(before, ca, normal)
            set_atom_coord(new_atom, after)
            displacements.append(float(np.linalg.norm(after - before)))
        mirrored.append(new_atom)
    sanity = _sanity_check(residue_atoms, mirrored, context_atoms or residue_atoms, max_displacement_A, max_bond_change_A, severe_overlap_A)
    if not sanity["accepted"]:
        return copy.deepcopy(residue_atoms), {
            "correction_applied": False,
            "n_atoms_mirrored": 0,
            "reason": "rollback_sanity_check",
            "mean_atom_displacement": 0.0,
            "max_atom_displacement": sanity["max_atom_displacement"],
            "max_bond_delta_A": sanity["max_bond_delta_A"],
            "new_severe_clashes": sanity["new_severe_clashes"],
            "reject_reasons": sanity["reject_reasons"],
            "bond_rows": sanity["bond_rows"],
        }
    return mirrored, {
        "correction_applied": bool(displacements),
        "n_atoms_mirrored": len(displacements),
        "reason": "mirrored_sidechain",
        "mean_atom_displacement": float(np.mean(displacements)) if displacements else 0.0,
        "max_atom_displacement": float(np.max(displacements)) if displacements else 0.0,
        "max_bond_delta_A": sanity["max_bond_delta_A"],
        "new_severe_clashes": sanity["new_severe_clashes"],
        "reject_reasons": [],
        "bond_rows": sanity["bond_rows"],
    }


def apply_local_mirrors(
    atoms: list[dict],
    correction_keys: set[tuple[str, int]],
) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    residue_updates: dict[tuple[str, int], list[dict]] = {}
    stats: dict[tuple[str, int], dict] = {}
    working = copy.deepcopy(atoms)
    for residue in iter_residues(atoms):
        key = (residue["chain_id"], int(residue["residue_index"]))
        if key in correction_keys:
            current_residue = next((r["atoms"] for r in iter_residues(working) if (r["chain_id"], int(r["residue_index"])) == key), residue["atoms"])
            mirrored, info = mirror_residue_sidechain(current_residue, working)
            residue_updates[key] = mirrored
            stats[key] = info
            if info.get("correction_applied"):
                by_key = {(a.get("chain"), int(a.get("resid", 0)), a.get("name")): a for a in working}
                for atom in mirrored:
                    by_key[(atom.get("chain"), int(atom.get("resid", 0)), atom.get("name"))] = copy.deepcopy(atom)
                working = list(by_key.values())

    counters: dict[tuple[str, int], int] = {key: 0 for key in residue_updates}
    out = []
    for atom in atoms:
        key = (atom.get("chain"), int(atom.get("resid", 0)))
        if key in residue_updates:
            out.append(copy.deepcopy(residue_updates[key][counters[key]]))
            counters[key] += 1
        else:
            out.append(copy.deepcopy(atom))
    return out, stats


def write_pdb(atoms: list[dict], path: str | Path, remarks: list[str] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for remark in remarks or []:
            f.write(f"REMARK   {remark}\n")
        for i, atom in enumerate(atoms, start=1):
            name = str(atom.get("name", ""))[:4]
            resname = str(atom.get("resname", "UNK"))[:3]
            chain = str(atom.get("chain", "A") or "A")[:1]
            resid = int(atom.get("resid", 0))
            x, y, z = float(atom["x"]), float(atom["y"]), float(atom["z"])
            occ = float(atom.get("occupancy", 1.0))
            bfac = float(atom.get("bfactor", 0.0))
            elem = str(atom.get("element", name[:1])).strip().upper()[:2]
            f.write(
                f"ATOM  {i:5d} {name:<4s} {resname:>3s} {chain:1s}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}\n"
            )
        f.write("END\n")


def write_cif(atoms: list[dict], path: str | Path, data_name: str = "chirality_corrected") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"data_{data_name}\n#\nloop_\n")
        headers = [
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_asym_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
        for h in headers:
            f.write(h + "\n")
        for i, atom in enumerate(atoms, start=1):
            chain = atom.get("label_asym_id") or atom.get("chain", "A")
            auth_chain = atom.get("auth_asym_id") or atom.get("chain", "A")
            resid = int(atom.get("label_seq_id") or atom.get("resid", 0))
            auth_resid = int(atom.get("auth_seq_id") or atom.get("resid", 0))
            f.write(
                "ATOM {id} {elem} {name} . {resname} {chain} 1 {resid} ? "
                "{x:.3f} {y:.3f} {z:.3f} {occ:.2f} {bfac:.2f} {auth_resid} {auth_chain} 1\n".format(
                    id=i,
                    elem=(atom.get("element") or atom.get("name", "X")[0]).upper(),
                    name=atom.get("name", "X"),
                    resname=atom.get("resname", "UNK"),
                    chain=chain,
                    resid=resid,
                    x=float(atom["x"]),
                    y=float(atom["y"]),
                    z=float(atom["z"]),
                    occ=float(atom.get("occupancy", 1.0)),
                    bfac=float(atom.get("bfactor", 0.0)),
                    auth_resid=auth_resid,
                    auth_chain=auth_chain,
                )
            )
        f.write("#\n")


def apply_error_code_file(
    atoms: list[dict],
    error_code_path: str | Path,
    correction_mode: str = "all",
    max_corrections: int | None = None,
) -> tuple[list[dict], dict, dict]:
    package = json.loads(Path(error_code_path).read_text(encoding="utf-8"))
    keys = correction_keys_from_error_package(package, correction_mode, max_corrections)
    corrected, stats = apply_local_mirrors(atoms, keys)
    quality = geometry_quality(atoms, corrected, keys)
    metadata = {
        "correction_method": "binary_chirality_error_code_plus_local_mirror",
        "correction_mode": correction_mode,
        "n_requested_corrections": len(keys),
        "n_applied_corrections": sum(1 for s in stats.values() if s.get("correction_applied")),
        "reference_used_for_correction": False,
        "target_chirality_source": "input_condition_or_CCD",
    }
    return corrected, {f"{k[0]}:{k[1]}": v for k, v in stats.items()}, {**metadata, **quality}


def classify_after_for_rows(before_atoms: list[dict], after_atoms: list[dict], rows: list[dict], correction_stats: dict) -> list[dict]:
    before_res = {(r["chain_id"], int(r["residue_index"])): r["atoms"] for r in iter_residues(before_atoms)}
    after_res = {(r["chain_id"], int(r["residue_index"])): r["atoms"] for r in iter_residues(after_atoms)}
    out = []
    for row in rows:
        key = (row["chain_id"], int(row["residue_index"]))
        before = classify_residue_chirality(before_res.get(key, []))
        after = classify_residue_chirality(after_res.get(key, []))
        stat = correction_stats.get(f"{key[0]}:{key[1]}", {})
        target = row.get("target_chirality")
        before_ok = before["chirality"] == target
        after_ok = after["chirality"] == target
        if before["chirality_status"] in {"unresolved", "ambiguous", "achiral"} or after["chirality_status"] in {"unresolved", "ambiguous", "achiral"}:
            status = "unresolved"
        elif _is_error_code_one(row.get("error_code")) and after_ok:
            status = "successfully_corrected"
        elif _is_error_code_one(row.get("error_code")) and not after_ok:
            status = "correction_failed"
        elif before_ok and after_ok:
            status = "correct_residue_preserved"
        elif before_ok and not after_ok:
            status = "correct_residue_damaged"
        else:
            status = "residual_error"
        out.append({
            "case_id": row["case_id"],
            "seed": row["seed"],
            "chain_id": key[0],
            "residue_index": key[1],
            "residue_name": row["residue_name"],
            "target_chirality": target,
            "chirality_before": before["chirality"] or before["chirality_status"],
            "chirality_after": after["chirality"] or after["chirality_status"],
            "signed_volume_before": before["signed_volume"],
            "signed_volume_after": after["signed_volume"],
            "error_code": row.get("error_code"),
            "correction_applied": bool(
                stat.get("correction_applied", _sidechain_changed(before_res.get(key, []), after_res.get(key, [])))
            ),
            "correction_status": status,
            "n_atoms_mirrored": int(stat.get("n_atoms_mirrored", 0)),
            "reject_reasons": ";".join(stat.get("reject_reasons") or []),
            "max_bond_delta_A": stat.get("max_bond_delta_A", ""),
            "new_severe_clashes": len(stat.get("new_severe_clashes") or []),
            "mean_atom_displacement": float(stat.get("mean_atom_displacement", 0.0)),
            "max_atom_displacement": float(stat.get("max_atom_displacement", 0.0)),
        })
    return out
