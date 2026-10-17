"""Deterministic chirality geometry utilities for AT-AF3 post-processing.

This module deliberately uses only the input structure coordinates and an
explicit target-chirality config. It never reads reference structures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

CHIRAL_ATOMS = ("N", "CA", "C", "CB")
FIXED_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
DEFAULT_EPSILON = 1e-6


def atom_coord(atom: dict) -> np.ndarray:
    return np.array([atom["x"], atom["y"], atom["z"]], dtype=float)


def set_atom_coord(atom: dict, coord: np.ndarray) -> None:
    atom["x"], atom["y"], atom["z"] = map(float, coord)


def residue_key(atom: dict) -> tuple:
    return (
        atom.get("chain", ""),
        int(atom.get("resid", 0)),
        atom.get("insertion_code", ""),
        atom.get("resname", ""),
    )


def residue_atom_map(atoms: Iterable[dict], chain: str, resid: int) -> dict[str, dict]:
    out = {}
    for atom in atoms:
        if atom.get("chain") == chain and int(atom.get("resid", -10**9)) == int(resid):
            out.setdefault(atom.get("name", ""), atom)
    return out


def iter_residues(atoms: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    first_seen: dict[tuple, int] = {}
    for i, atom in enumerate(atoms):
        key = residue_key(atom)
        grouped.setdefault(key, []).append(atom)
        first_seen.setdefault(key, i)
    residues = []
    for key in sorted(grouped, key=lambda k: first_seen[k]):
        chain, resid, insertion_code, resname = key
        residues.append({
            "chain_id": chain,
            "residue_index": int(resid),
            "insertion_code": insertion_code,
            "residue_name": resname,
            "atoms": grouped[key],
        })
    return residues


def signed_chiral_volume(atom_map: dict[str, dict]) -> float:
    coords = {name: atom_coord(atom_map[name]) for name in CHIRAL_ATOMS}
    return float(np.dot(np.cross(coords["N"] - coords["CA"], coords["C"] - coords["CA"]), coords["CB"] - coords["CA"]))


def _synthetic_ala_atoms(cb_z: float) -> list[dict]:
    coords = {
        "N": np.array([1.0, 0.0, 0.0]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0]),
        "O": np.array([0.0, 1.5, 0.0]),
        "CB": np.array([0.0, 0.0, cb_z]),
    }
    return [
        {"serial": i + 1, "name": name, "resname": "ALA", "chain": "A", "resid": 1,
         "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]), "element": name[0]}
        for i, (name, xyz) in enumerate(coords.items())
    ]


def calibrate_chirality_sign() -> dict:
    l_atoms = _synthetic_ala_atoms(1.0)
    d_atoms = _synthetic_ala_atoms(-1.0)
    l_volume = signed_chiral_volume(residue_atom_map(l_atoms, "A", 1))
    d_volume = signed_chiral_volume(residue_atom_map(d_atoms, "A", 1))
    if not (l_volume > 0.0 and d_volume < 0.0):
        raise RuntimeError(
            f"Chirality sign calibration failed: L-Ala={l_volume}, D-Ala={d_volume}"
        )
    return {
        "atom_order": ["N", "C", "CB"],
        "signed_volume_formula": "dot(cross(N - CA, C - CA), CB - CA)",
        "positive_sign": "L",
        "negative_sign": "D",
        "calibration_status": "verified",
        "L_ala_signed_volume": l_volume,
        "D_ala_signed_volume": d_volume,
        "reference_used_for_calibration": False,
    }


def ensure_chirality_sign_mapping(path: str | Path | None = None) -> dict:
    mapping = calibrate_chirality_sign()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return mapping


def classify_residue_chirality(residue_atoms: list[dict], epsilon: float = DEFAULT_EPSILON) -> dict:
    resname = residue_atoms[0].get("resname", "") if residue_atoms else ""
    if resname == "GLY":
        return {
            "chirality": "achiral",
            "chirality_status": "achiral",
            "signed_volume": None,
            "reason": "glycine_no_cb",
        }
    atom_map = {a.get("name", ""): a for a in residue_atoms}
    missing = [name for name in CHIRAL_ATOMS if name not in atom_map]
    if missing:
        return {
            "chirality": None,
            "chirality_status": "unresolved",
            "signed_volume": None,
            "reason": "missing_atoms:" + ",".join(missing),
        }
    volume = signed_chiral_volume(atom_map)
    if abs(volume) < epsilon:
        return {
            "chirality": None,
            "chirality_status": "ambiguous",
            "signed_volume": volume,
            "reason": "ambiguous_near_planar",
        }
    return {
        "chirality": "L" if volume > 0 else "D",
        "chirality_status": "resolved",
        "signed_volume": volume,
        "reason": "computed",
    }


def reflect_point_across_plane(point: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(normal))
    if denom < 1e-12:
        raise ValueError("Cannot mirror across a degenerate N-CA-C plane")
    unit = normal / denom
    return point - 2.0 * float(np.dot(point - plane_point, unit)) * unit


def mirrorable_atom(atom: dict) -> bool:
    name = atom.get("name", "")
    element = str(atom.get("element", "")).upper()
    if name in FIXED_BACKBONE_ATOMS:
        return False
    if name.startswith("H") or element == "H":
        return True
    return True


def clash_count(atoms: list[dict], threshold: float = 2.0) -> int:
    count = 0
    coords = [atom_coord(a) for a in atoms]
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if residue_key(atoms[i]) == residue_key(atoms[j]):
                continue
            if atoms[i].get("chain") == atoms[j].get("chain") and abs(int(atoms[i].get("resid", 0)) - int(atoms[j].get("resid", 0))) <= 1:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) < threshold:
                count += 1
    return count


def matched_atom_pairs(before: list[dict], after: list[dict], selector=None) -> list[tuple[dict, dict]]:
    after_map = {
        (a.get("chain"), a.get("resid"), a.get("insertion_code", ""), a.get("name")): a
        for a in after
    }
    pairs = []
    for atom in before:
        if selector and not selector(atom):
            continue
        key = (atom.get("chain"), atom.get("resid"), atom.get("insertion_code", ""), atom.get("name"))
        if key in after_map:
            pairs.append((atom, after_map[key]))
    return pairs


def rmsd_for_pairs(pairs: list[tuple[dict, dict]]) -> float:
    if not pairs:
        return math.nan
    diffs = [atom_coord(a) - atom_coord(b) for a, b in pairs]
    return float(np.sqrt(np.mean([float(np.dot(d, d)) for d in diffs])))


def bond_length_anomaly_count(atoms: list[dict]) -> int:
    expected = {
        ("N", "CA"): (1.20, 1.65),
        ("CA", "C"): (1.25, 1.70),
        ("C", "O"): (1.05, 1.45),
        ("CA", "CB"): (1.25, 1.75),
    }
    n_bad = 0
    for residue in iter_residues(atoms):
        amap = {a.get("name"): a for a in residue["atoms"]}
        for (a, b), (low, high) in expected.items():
            if a in amap and b in amap:
                dist = float(np.linalg.norm(atom_coord(amap[a]) - atom_coord(amap[b])))
                if dist < low or dist > high:
                    n_bad += 1
    return n_bad


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-12:
        return math.nan
    cosv = max(-1.0, min(1.0, float(np.dot(v1, v2) / denom)))
    return float(np.degrees(np.arccos(cosv)))


def bond_angle_anomaly_count(atoms: list[dict]) -> int:
    triples = (("N", "CA", "C"), ("N", "CA", "CB"), ("C", "CA", "CB"))
    n_bad = 0
    for residue in iter_residues(atoms):
        amap = {a.get("name"): a for a in residue["atoms"]}
        for names in triples:
            if all(name in amap for name in names):
                angle = _angle_degrees(*(atom_coord(amap[name]) for name in names))
                if not math.isfinite(angle) or angle < 80.0 or angle > 140.0:
                    n_bad += 1
    return n_bad


def geometry_quality(before: list[dict], after: list[dict], corrected_keys: set[tuple] | None = None) -> dict:
    corrected_keys = corrected_keys or set()
    backbone = lambda a: a.get("name") in FIXED_BACKBONE_ATOMS
    ca = lambda a: a.get("name") == "CA"
    corrected = lambda a: (a.get("chain"), int(a.get("resid", 0))) in corrected_keys
    uncorrected = lambda a: (a.get("chain"), int(a.get("resid", 0))) not in corrected_keys
    before_clashes = clash_count(before)
    after_clashes = clash_count(after)
    return {
        "backbone_rmsd": rmsd_for_pairs(matched_atom_pairs(before, after, backbone)),
        "ca_rmsd": rmsd_for_pairs(matched_atom_pairs(before, after, ca)),
        "all_atom_rmsd": rmsd_for_pairs(matched_atom_pairs(before, after)),
        "corrected_residue_local_rmsd": rmsd_for_pairs(matched_atom_pairs(before, after, corrected)),
        "uncorrected_residue_rmsd": rmsd_for_pairs(matched_atom_pairs(before, after, uncorrected)),
        "clash_count_before": before_clashes,
        "clash_count_after": after_clashes,
        "new_clash_count": max(0, after_clashes - before_clashes),
        "clash_rate_delta": (after_clashes - before_clashes) / max(len(after), 1),
        "bond_length_anomalies_before": bond_length_anomaly_count(before),
        "bond_length_anomalies_after": bond_length_anomaly_count(after),
        "bond_angle_anomalies_before": bond_angle_anomaly_count(before),
        "bond_angle_anomalies_after": bond_angle_anomaly_count(after),
        "backbone_moved_warning": rmsd_for_pairs(matched_atom_pairs(before, after, backbone)) > 1e-6,
    }
