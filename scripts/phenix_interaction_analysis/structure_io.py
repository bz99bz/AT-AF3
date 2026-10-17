"""Structure ingestion with iotbx/cctbx, into one harmonised model.

A structure (GT mmCIF/PDB, an AF3 model, an AT-AF3 model) is read once into
:class:`Structure` -> :class:`Chain` -> :class:`Residue` -> :class:`Atom`.  The residue key keeps
the author sequence number **and the insertion code** (``(chain, seq, icode)``), so nothing is
renumbered and nothing is merged.  Residues are typed with :class:`~.chem.ChemResolver`, which is
also what turns a D-CCD name (``DAL``) into its parent chemistry (``ALA``) while leaving the
coordinates untouched.

Alternate conformers are handled by one documented policy: keep the highest-occupancy altloc of
each atom group (ties broken A, B, C, ...) and record every dropped atom.  ``keep_all`` is
available for callers that need the raw file, but the analysis pipeline uses ``best``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol as P
from .chem import ChemResolver


@dataclass
class Atom:
    name: str
    element: str
    xyz: tuple
    occ: float
    altloc: str
    is_hydrogen: bool
    record: str = "ATOM"


@dataclass
class Residue:
    chain: str
    seq: str
    icode: str
    resname: str
    chem: object
    is_het: bool
    atoms: list = field(default_factory=list)

    @property
    def token(self) -> str:
        return f"{self.chain}:{self.seq}{self.icode}".rstrip()

    @property
    def is_water(self) -> bool:
        return bool(getattr(self.chem, "is_water", False))

    @property
    def is_polymer(self) -> bool:
        return bool(getattr(self.chem, "is_polymer", False))

    def heavy_atoms(self) -> list:
        return [a for a in self.atoms if not a.is_hydrogen]

    def atom(self, name: str):
        for a in self.atoms:
            if a.name == name:
                return a
        return None


@dataclass
class Chain:
    id: str
    residues: list = field(default_factory=list)

    def polymer_residues(self) -> list:
        return [r for r in self.residues if r.is_polymer]


@dataclass
class Structure:
    name: str
    source_path: str
    chains: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- accessors ----------------------------------------------------------
    def chain(self, chain_id: str):
        for ch in self.chains:
            if ch.id == chain_id:
                return ch
        return None

    def chain_ids(self) -> list:
        return [c.id for c in self.chains]

    def residues(self):
        for ch in self.chains:
            for r in ch.residues:
                yield r

    def polymer_residues(self):
        for ch in self.chains:
            for r in ch.residues:
                if r.is_polymer:
                    yield r

    def water_residues(self):
        for ch in self.chains:
            for r in ch.residues:
                if r.is_water:
                    yield r

    def resnames(self) -> set:
        return {r.resname for r in self.residues()}

    def atom_count(self) -> int:
        return sum(len(r.atoms) for r in self.residues())

    def heavy_atom_count(self) -> int:
        return sum(len(r.heavy_atoms()) for r in self.residues())


_TWO_LETTER_ELEMENTS = frozenset({
    "CL", "BR", "NA", "MG", "ZN", "FE", "MN", "CA", "CU", "CO", "NI", "CD", "HG", "SE",
    "LI", "AL", "SI", "BA", "SR", "CS", "RB", "MO", "PT", "AU", "AG", "IR", "OS", "RE",
    "TL", "PB", "BI", "SN", "SB", "TE", "GA", "IN", "GE", "CR", "V", "W", "HF", "TA",
})


def _guess_element_from_name(name: str, resname: str = "") -> str:
    nm = (name or "").strip()
    i = 0
    while i < len(nm) and (nm[i].isdigit() or nm[i] == " "):
        i += 1
    nm = nm[i:]
    if not nm:
        return ""
    up = nm.upper()
    if len(up) >= 2 and up[:2] in _TWO_LETTER_ELEMENTS and (len(up) == 2 or not up[2].isalpha()):
        return up[:2]
    return up[0]


def _element_of(atom, resname: str) -> str:
    el = (getattr(atom, "element", "") or "").strip().upper()
    if el:
        return el
    return _guess_element_from_name(atom.name, resname)


def _build_hierarchy(path: Path):
    """Return an iotbx hierarchy for a PDB/mmCIF file (PDB first, mmCIF as fallback)."""
    from iotbx import pdb as iotbx_pdb
    try:
        return iotbx_pdb.input(file_name=str(path)).construct_hierarchy()
    except Exception:
        from iotbx import cif as iotbx_cif
        reader = iotbx_cif.reader(file_path=str(path))
        cif_model = reader.model()
        return iotbx_pdb.hierarchy.input(pdb_string=cif_model.as_pdb_string()).hierarchy


def read_structure(path, name: str = "structure", resolver: ChemResolver | None = None,
                   altloc_policy: str = "best"):
    """Read one structure; return ``(Structure, audit_dict)``.

    ``altloc_policy`` is ``best`` (pipeline default) or ``all``.
    """
    path = Path(path)
    resolver = resolver or ChemResolver()
    audit = {"models_in_file": 0, "hydrogens_in_file": 0, "alternate_locator_residues": 0,
             "alternate_locator_atoms_dropped": 0, "altloc_policy": altloc_policy}
    if not path.exists():
        return Structure(name=name, source_path=str(path)), audit
    hierarchy = _build_hierarchy(path)
    models = list(hierarchy.models())
    audit["models_in_file"] = len(models)
    structure = Structure(name=name, source_path=str(path))
    if not models:
        return structure, audit
    used = models[0]
    seen_resnames = set()
    hydrogens_in_file = 0
    altloc_residues = set()
    dropped = 0
    by_id = {}
    order = []
    for ch in used.chains():
        chain = Chain(id=ch.id)
        for rg in ch.residue_groups():
            seq = str(rg.resseq).strip()
            icode = (str(rg.icode).strip() if rg.icode else "")
            groups = list(rg.atom_groups())
            if altloc_policy == "best":
                by_res = {}
                for ag in groups:
                    by_res.setdefault(ag.resname, []).append(ag)
                kept = []
                for resname, lst in by_res.items():
                    if len(lst) == 1:
                        kept.append(lst[0])
                        continue
                    best = max(lst, key=lambda g: sum(float(a.occ) for a in g.atoms()))
                    kept.append(best)
                    altloc_residues.add((ch.id, seq, icode))
                    for g in lst:
                        if g is best:
                            continue
                        for a in g.atoms():
                            dropped += 1
                            if (a.element or "").strip().upper() in ("H", "D"):
                                hydrogens_in_file += 1
            else:
                kept = groups
            for ag in kept:
                resname = ag.resname.strip()
                info = resolver.resolve(resname)
                seen_resnames.add(resname)
                residue = Residue(chain=ch.id, seq=seq, icode=icode, resname=resname,
                                  chem=info, is_het=not info.is_polymer)
                for at in ag.atoms():
                    el = _element_of(at, resname)
                    if el in ("H", "D"):
                        hydrogens_in_file += 1
                    xyz = (round(float(at.xyz[0]), 4), round(float(at.xyz[1]), 4),
                           round(float(at.xyz[2]), 4))
                    residue.atoms.append(Atom(
                        name=at.name.strip(), element=el, xyz=xyz, occ=round(float(at.occ), 2),
                        altloc=(ag.altloc or "").strip(), is_hydrogen=el in ("H", "D"),
                        record="HETATM" if residue.is_het else "ATOM"))
                chain.residues.append(residue)
        if ch.id not in by_id:
            by_id[ch.id] = Chain(id=ch.id, residues=[])
            order.append(ch.id)
        by_id[ch.id].residues.extend(chain.residues)
    structure.chains = [by_id[cid] for cid in order]
    audit["hydrogens_in_file"] = hydrogens_in_file
    audit["alternate_locator_residues"] = len(altloc_residues)
    audit["alternate_locator_atoms_dropped"] = dropped
    audit["resnames"] = sorted(seen_resnames)
    audit["atoms"] = structure.atom_count()
    audit["heavy_atoms"] = structure.heavy_atom_count()
    structure.meta = audit
    return structure, audit

