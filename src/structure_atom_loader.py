"""Structure atom loading with mmCIF-first residue metadata.

This loader preserves label/auth chain and residue identifiers so clean-v4.2
mapping does not rely only on lossy PDB chain + residue number fields.
"""

from __future__ import annotations

import shlex
from pathlib import Path


def _clean(value: str) -> str:
    return "" if value in {".", "?"} else str(value)


def _to_int(value: str):
    try:
        return int(float(value))
    except Exception:
        return None


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _atom_site_loop(cif_path: Path) -> tuple[list[str], list[list[str]]]:
    lines = cif_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        j = i + 1
        headers = []
        while j < len(lines) and lines[j].startswith("_atom_site."):
            headers.append(lines[j].strip())
            j += 1
        if not headers:
            continue
        rows = []
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            rows.append(shlex.split(stripped))
            j += 1
        return headers, rows
    raise ValueError(f"No _atom_site loop found in {cif_path}")


def _pick_altloc(atoms: list[dict]) -> list[dict]:
    best = {}
    for atom in atoms:
        key = (
            atom.get("model_index", 1),
            atom.get("label_asym_id") or atom.get("chain"),
            atom.get("label_seq_id"),
            atom.get("auth_seq_id"),
            atom.get("insertion_code", ""),
            atom.get("name"),
        )
        alt = atom.get("altloc", "")
        rank = (float(atom.get("occupancy", 0.0)), 1 if alt in {"", "A"} else 0, -ord(alt[:1] or "Z"))
        if key not in best or rank > best[key][0]:
            best[key] = (rank, atom)
    return [v[1] for v in best.values()]


def parse_cif_atoms(path: str | Path, prefer_label_ids: bool = True, model_index: int = 1) -> list[dict]:
    cif_path = Path(path)
    headers, rows = _atom_site_loop(cif_path)
    idx = {h: i for i, h in enumerate(headers)}
    required = [
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    missing = [h for h in required if h not in idx]
    if missing:
        raise ValueError(f"{cif_path}: missing fields {missing}")
    atoms = []
    for serial, row in enumerate(rows, start=1):
        if len(row) < len(headers):
            continue
        pdbx_model = _to_int(row[idx.get("_atom_site.pdbx_PDB_model_num", -1)]) if "_atom_site.pdbx_PDB_model_num" in idx else 1
        if pdbx_model is not None and int(pdbx_model) != int(model_index):
            continue
        label_seq = _to_int(row[idx["_atom_site.label_seq_id"]]) if "_atom_site.label_seq_id" in idx else None
        auth_seq = _to_int(row[idx["_atom_site.auth_seq_id"]]) if "_atom_site.auth_seq_id" in idx else None
        label_chain = _clean(row[idx["_atom_site.label_asym_id"]])
        auth_chain = _clean(row[idx["_atom_site.auth_asym_id"]]) if "_atom_site.auth_asym_id" in idx else label_chain
        resid = label_seq if prefer_label_ids and label_seq is not None else auth_seq
        if resid is None:
            continue
        atom = {
            "serial": serial,
            "name": _clean(row[idx["_atom_site.label_atom_id"]]),
            "resname": _clean(row[idx["_atom_site.label_comp_id"]]),
            "chain": label_chain if prefer_label_ids else auth_chain,
            "resid": int(resid),
            "x": _to_float(row[idx["_atom_site.Cartn_x"]]),
            "y": _to_float(row[idx["_atom_site.Cartn_y"]]),
            "z": _to_float(row[idx["_atom_site.Cartn_z"]]),
            "occupancy": _to_float(row[idx["_atom_site.occupancy"]], 1.0) if "_atom_site.occupancy" in idx else 1.0,
            "bfactor": _to_float(row[idx["_atom_site.B_iso_or_equiv"]], 0.0) if "_atom_site.B_iso_or_equiv" in idx else 0.0,
            "element": _clean(row[idx["_atom_site.type_symbol"]]).upper() if "_atom_site.type_symbol" in idx else "",
            "label_asym_id": label_chain,
            "auth_asym_id": auth_chain,
            "entity_id": _clean(row[idx["_atom_site.label_entity_id"]]) if "_atom_site.label_entity_id" in idx else "",
            "label_seq_id": label_seq,
            "auth_seq_id": auth_seq,
            "insertion_code": _clean(row[idx["_atom_site.pdbx_PDB_ins_code"]]) if "_atom_site.pdbx_PDB_ins_code" in idx else "",
            "altloc": _clean(row[idx["_atom_site.label_alt_id"]]) if "_atom_site.label_alt_id" in idx else "",
            "model_index": pdbx_model or 1,
            "coordinate_source": "cif",
        }
        atoms.append(atom)
    return _pick_altloc(atoms)


def parse_pdb_atoms(path: str | Path) -> list[dict]:
    atoms = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for serial, line in enumerate(f, start=1):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resid = _to_int(line[22:26].strip())
            if resid is None:
                continue
            chain = line[21].strip() or "A"
            atoms.append({
                "serial": serial,
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": chain,
                "resid": int(resid),
                "x": _to_float(line[30:38]),
                "y": _to_float(line[38:46]),
                "z": _to_float(line[46:54]),
                "occupancy": _to_float(line[54:60], 1.0),
                "bfactor": _to_float(line[60:66], 0.0),
                "element": (line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[:1]).upper(),
                "label_asym_id": chain,
                "auth_asym_id": chain,
                "entity_id": "",
                "label_seq_id": int(resid),
                "auth_seq_id": int(resid),
                "insertion_code": line[26].strip(),
                "altloc": line[16].strip(),
                "model_index": 1,
                "coordinate_source": "pdb",
            })
    return _pick_altloc(atoms)


def load_structure_atoms(path: str | Path, prefer_cif: bool = True) -> tuple[list[dict], dict]:
    p = Path(path)
    if not p.exists():
        return [], {"coordinate_source": "", "parse_success": False, "error": "source_structure_missing"}
    try:
        if p.suffix.lower() in {".cif", ".mmcif"}:
            atoms = parse_cif_atoms(p)
            return atoms, {"coordinate_source": "cif", "parse_success": True, "altloc_policy": "highest_occupancy_then_A"}
        atoms = parse_pdb_atoms(p)
        return atoms, {"coordinate_source": "pdb", "parse_success": True, "altloc_policy": "highest_occupancy_then_A"}
    except Exception as exc:
        return [], {"coordinate_source": p.suffix.lower().lstrip("."), "parse_success": False, "error": str(exc)}
