"""Residue/atom chemistry used for interaction typing and D-residue recognition.

A residue is resolved to the chemistry of its **parent** L amino acid (so a D-CCD residue such
as ``DAL`` is typed as ``ALA`` without imposing L geometry).  The parent is taken, in order, from

* the curated D-CCD -> L table (covers every D residue in this benchmark), then
* the wwPDB CCD entry's ``one_letter_code`` / ``mon_nstd_parent_comp_id`` (so a modified
  residue is typed as its parent when the CCD declares one), then
* the residue name itself when it is one of the 20 standard amino acids.

Anything left over is reported as **unsupported chemistry**; typing is not inferred without
evidence.  Atom
roles (donor / acceptor / anion / cation / disulfide sulfur / hydrophobic) come from the frozen
tables in :mod:`.protocol`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol as P

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
# D-CCD -> L parent (curated; the CCD one-letter code is used as a cross-check when present).
D_TO_L = {
    "DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS", "DGL": "GLU", "DGN": "GLN",
    "DSG": "ASN", "DSN": "SER", "DHI": "HIS", "DIL": "ILE", "DLE": "LEU", "DLY": "LYS",
    "MED": "MET", "DPN": "PHE", "DPR": "PRO", "DTH": "THR", "DTR": "TRP", "DTY": "TYR",
    "DVA": "VAL",
}
# modified residues with no one-letter code whose parent is a protocol choice.
MODIFIED_PARENT_FALLBACK = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "CSO": "CYS", "CME": "CYS", "MLY": "LYS",
    "ORN": "LYS", "NLE": "LEU", "SAR": "GLY", "AIB": "ALA", "MLU": "LEU", "HYP": "PRO",
}

ROLE_DONOR = "donor"
ROLE_ACCEPTOR = "acceptor"
ROLE_ANION = "anion"
ROLE_CATION = "cation"
ROLE_CATION_EXTENDED = "cation_extended"
ROLE_DISULFIDE = "disulfide_sulfur"
ROLE_HYDROPHOBIC = "hydrophobic_capable"


@dataclass
class ChemInfo:
    resname: str
    chem_class: str = ""
    parent: str = ""
    is_D: bool = False
    is_polymer: bool = False
    is_supported: bool = False
    is_water: bool = False
    source: str = ""
    note: str = ""


@dataclass
class ChemResolver:
    """Resolve residue names to chemistry; optionally backed by the CCD cache."""

    ccd_cache: Path | None = None
    _components: dict = field(default_factory=dict, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)
    _seen: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        if self.ccd_cache and Path(self.ccd_cache).exists():
            data = json.loads(Path(self.ccd_cache).read_text())
            self._components = data.get("components", {}) or {}

    def resolve(self, resname: str) -> ChemInfo:
        resname = (resname or "").strip().upper()
        if resname in self._cache:
            self._seen.add(resname)
            return self._cache[resname]
        info = self._resolve(resname)
        self._cache[resname] = info
        self._seen.add(resname)
        return info

    def _resolve(self, resname: str) -> ChemInfo:
        if resname in P.WATER_NAMES:
            return ChemInfo(resname, is_water=True, is_polymer=False, is_supported=True,
                            source="water", note="solvent")
        if resname in P.STANDARD_AA:
            return ChemInfo(resname, chem_class=resname, parent=resname, is_polymer=True,
                            is_supported=True, source="standard_aa")
        if resname in D_TO_L:
            return ChemInfo(resname, chem_class=D_TO_L[resname], parent=D_TO_L[resname],
                            is_D=True, is_polymer=True, is_supported=True,
                            source="d_to_l_table", note="D residue typed as parent chemistry")
        comp = self._components.get(resname)
        if comp is not None:
            one = (comp.get("one_letter_code") or "").strip().upper()
            parent = ONE_TO_THREE.get(one, "")
            is_d = str(comp.get("comp_type", "")).upper().startswith("D-")
            if not parent and is_d and resname.startswith("D"):
                parent = resname[1:]
            is_poly = "PEPTIDE" in str(comp.get("comp_type", "")).upper()
            supported = parent in P.STANDARD_AA
            return ChemInfo(resname, chem_class=parent if supported else "", parent=parent,
                            is_D=is_d, is_polymer=is_poly, is_supported=supported,
                            source="ccd", note="" if supported else "CCD parent not a standard AA")
        if resname in MODIFIED_PARENT_FALLBACK:
            parent = MODIFIED_PARENT_FALLBACK[resname]
            return ChemInfo(resname, chem_class=parent, parent=parent, is_polymer=True,
                            is_supported=True, source="modified_fallback",
                            note="parent is a protocol choice")
        return ChemInfo(resname, is_polymer=False, is_supported=False, source="unresolved",
                        note="not in CCD cache, standard set or fallback tables")

    def atom_roles(self, resname: str, atom_name: str) -> set:
        info = self.resolve(resname)
        if info.is_water:
            return {ROLE_DONOR, ROLE_ACCEPTOR}
        if not info.is_supported:
            return set()
        cls = info.chem_class
        name = atom_name.strip()
        roles = set()
        if name == "N":
            roles.add(ROLE_DONOR)
        if name in ("O", "OXT"):
            roles.add(ROLE_ACCEPTOR)
        if name in P.SIDE_CHAIN_DONORS.get(cls, ()):
            roles.add(ROLE_DONOR)
        if name in P.SIDE_CHAIN_ACCEPTORS.get(cls, ()):
            roles.add(ROLE_ACCEPTOR)
        if name in P.SIDE_CHAIN_ANIONS.get(cls, ()):
            roles.add(ROLE_ANION)
        if name in P.SIDE_CHAIN_CATIONS.get(cls, ()):
            roles.add(ROLE_CATION)
        if name in P.SIDE_CHAIN_CATIONS_EXTENDED.get(cls, ()):
            roles.add(ROLE_CATION_EXTENDED)
        if name == "SG" and cls == "CYS":
            roles.add(ROLE_DISULFIDE)
        if cls in P.HYDROPHOBIC_RESIDUES and name not in ("N", "O", "OXT"):
            roles.add(ROLE_HYDROPHOBIC)
        return roles

    def audit_rows(self) -> list:
        rows = []
        for resname in sorted(self._seen):
            info = self._cache[resname]
            rows.append({
                "resname": resname, "chem_class": info.chem_class, "parent": info.parent,
                "is_D": "yes" if info.is_D else "no",
                "is_polymer": "yes" if info.is_polymer else "no",
                "is_water": "yes" if info.is_water else "no",
                "supported": "yes" if info.is_supported else "no",
                "source": info.source, "note": info.note,
            })
        return rows
