"""Hydrogen placement with Phenix ``reduce`` and a heavy-atom preservation record.

The analysis needs hydrogen positions only for the donor-H...acceptor angle of a hydrogen bond.
Hydrogens are placed on a **copy** of the structure by the installed Phenix ``reduce``
(``-BUILD -NOFLIP``), and the copy is then compared atom-by-atom with the input:

* every input heavy atom must survive with the same name and stay within
  :data:`~.protocol.RETENTION_DISTANCE_TOLERANCE` - here a tight 0.005 A - otherwise the
  hydrogen set is declared *not usable* and the analysis falls back to the separately labelled
  ``polar_contact_heavy_atom`` class (distance only);
* D-CCD residues are renamed to their L parent **for the reduce input only** (reduce has no
  chemistry for ``DAL`` and would otherwise leave the residue unprotonated); the heavy atoms do
  not move, so the geometry - including the donor-H direction - is unchanged.  The rename is
  recorded in ``preparation_audit.csv``.

Nothing is minimised, rebuilt, re-flipped (``-NOFLIP``) or repaired, and no heavy atom is added.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import protocol as P
from .structure_io import read_structure

HEAVY_SHIFT_TOLERANCE = 0.005


def _atom_name_field(name: str) -> str:
    if len(name) >= 4:
        return name[:4]
    if len(name) == 3:
        return " " + name
    return " " + name + " " * (3 - len(name))


def structure_to_pdb_lines(structure, rename: dict | None = None) -> list:
    """Serialise the harmonised model back to PDB, heavy atoms only."""
    rename = rename or {}
    lines = []
    serial = 0
    for chain in structure.chains:
        for res in chain.residues:
            resname = rename.get(res.resname, res.resname)
            for atom in res.heavy_atoms():
                serial += 1
                rec = "HETATM" if res.is_het else "ATOM"
                seq = res.seq if len(res.seq) <= 4 else res.seq[-4:]
                x, y, z = atom.xyz
                lines.append(
                    "%-6s%5d %-4s%1s%3s %1s%4s%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
                    % (rec, serial, _atom_name_field(atom.name), atom.altloc or " ",
                       resname, chain.id, seq, res.icode or " ", x, y, z, atom.occ, 0.0,
                       atom.element))
        lines.append("TER")
    lines.append("END")
    return lines


def reduce_version(reduce_tool: str) -> str:
    if not reduce_tool or not Path(reduce_tool).exists():
        return "unavailable"
    try:
        out = subprocess.run([reduce_tool, "-h"], capture_output=True, text=True, timeout=60)
        text = (out.stdout or "") + (out.stderr or "")
        for line in text.splitlines():
            if line.strip().lower().startswith("reduce:"):
                return line.strip()
    except Exception:
        pass
    return "available (version string not parsed)"


def _write_lines(path: Path, lines: list) -> None:
    path.write_text("\n".join(lines) + "\n")


def hydrogenate(structure, reduce_tool: str, workdir: Path, tag: str, resolver):
    """Place hydrogens on a copy; return ``(reduced_structure, audit, placed_hydrogens)``."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    audit = {
        "reduce_tool": reduce_tool or P.NA,
        "reduce_version": reduce_version(reduce_tool),
        "protonation_policy": P.PROTONATION_POLICY,
        "altered_resnames": "",
        "heavy_atoms_input": structure.heavy_atom_count(),
        "heavy_atoms_hydrogenated": 0,
        "heavy_atoms_matched": 0,
        "heavy_atoms_missing": 0,
        "heavy_atoms_added": 0,
        "max_heavy_atom_shift": P.NA,
        "hydrogens_added": 0,
        "reduce_status": "not_run",
        "heavy_atoms_unchanged": "no",
        "usable": False,
        "reason": "",
    }
    placed: dict = {}
    if not reduce_tool or not Path(reduce_tool).exists():
        audit["reduce_status"] = "unavailable"
        audit["reason"] = "reduce tool not configured or missing"
        return structure, audit, placed

    rename = {}
    for res in structure.residues():
        if getattr(res.chem, "is_D", False) and res.chem.parent:
            rename[res.resname] = res.chem.parent
    audit["altered_resnames"] = ";".join(f"{k}->{v}" for k, v in sorted(rename.items()))

    inp = workdir / f"{tag}_heavy.pdb"
    out = workdir / f"{tag}_reduced.pdb"
    _write_lines(inp, structure_to_pdb_lines(structure, rename=rename))
    cmd = [reduce_tool, "-BUILD", "-NOFLIP", "-QUIET", str(inp)]
    try:
        with open(out, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True, timeout=600)
    except Exception as exc:  # pragma: no cover - defensive
        audit["reduce_status"] = "failed_to_run"
        audit["reason"] = str(exc)
        return structure, audit, placed
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        audit["reduce_status"] = f"failed (exit {proc.returncode})"
        audit["reason"] = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "empty output"
        return structure, audit, placed

    reduced, _ = read_structure(out, name=structure.name + "_reduced", resolver=resolver)
    audit["reduce_status"] = "ok"
    audit["heavy_atoms_hydrogenated"] = reduced.heavy_atom_count()
    audit["hydrogens_added"] = reduced.atom_count() - reduced.heavy_atom_count()

    # -- heavy-atom preservation ------------------------------------------
    def heavy_map(struct):
        m = {}
        for ch in struct.chains:
            for res in ch.residues:
                if res.is_water:
                    continue
                for a in res.heavy_atoms():
                    m[(ch.id, res.seq, res.icode, a.name)] = a.xyz
        return m

    src = heavy_map(structure)
    got = heavy_map(reduced)
    matched = missing = added = 0
    max_shift = 0.0
    for key, xyz in src.items():
        other = got.get(key)
        if other is None:
            missing += 1
            continue
        matched += 1
        d = max(abs(xyz[i] - other[i]) for i in range(3))
        max_shift = max(max_shift, d)
    for key in got:
        if key not in src:
            added += 1
    audit["heavy_atoms_matched"] = matched
    audit["heavy_atoms_missing"] = missing
    audit["heavy_atoms_added"] = added
    audit["max_heavy_atom_shift"] = round(max_shift, 4)
    usable = (missing == 0 and max_shift <= HEAVY_SHIFT_TOLERANCE)
    audit["heavy_atoms_unchanged"] = "yes" if usable else "no"
    audit["usable"] = usable
    if not usable:
        if missing:
            audit["reason"] = f"{missing} heavy atoms missing from the reduce output"
        else:
            audit["reason"] = f"largest heavy-atom shift {max_shift:.3f} A"

    # -- donor hydrogen inventory -----------------------------------------
    for ch in reduced.chains:
        for res in ch.residues:
            heavies = res.heavy_atoms()
            for h in res.atoms:
                if not h.is_hydrogen:
                    continue
                best, attached = 9.9, None
                for hv in heavies:
                    d = sum((h.xyz[i] - hv.xyz[i]) ** 2 for i in range(3)) ** 0.5
                    if d < best:
                        best, attached = d, hv.name
                if attached is not None and best <= P.HBOND_H_LENGTH + 0.2:
                    placed.setdefault((ch.id, res.seq, res.icode, attached), []).append(
                        (h.name, h.xyz))
    return reduced, audit, placed