"""The frozen detector and the common interaction record.

One detector, frozen settings from :mod:`.protocol`, is run over every structure.  Its output is
the same :class:`InteractionRecord` for GT, AF3, uncorrected and corrected AT-AF3 - so the rows
can be compared directly.  Each record keeps the identifiers (target / method / seed / residue
tokens with insertion codes), the type (class + scope + subset), the partners and atoms, the
geometry (distance and donor-H...acceptor angle) and an evaluation status.

Atom roles come from :mod:`.chem` (donor / acceptor / anion / cation / disulfide sulfur /
hydrophobic); water and metal atoms are typed from the element set.  Hydrogen-bond angles use the
donor hydrogens placed by :mod:`.prepare`; when a donor has no hydrogen the contact is reported
as the separately labelled ``polar_contact_heavy_atom`` class and is never merged into ``hbond``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import protocol as P
from . import chem as C
from . import mapping as M


# ---------------------------------------------------------------------------
#  neighbour grid
# ---------------------------------------------------------------------------
class Grid:
    """Uniform-cell neighbour search over a list of ``(payload, xyz)`` items."""

    def __init__(self, items, cell: float = 6.0):
        self.cell = cell
        self.cells: dict = {}
        for idx, (_payload, xyz) in enumerate(items):
            key = (int(math.floor(xyz[0] / cell)), int(math.floor(xyz[1] / cell)),
                   int(math.floor(xyz[2] / cell)))
            self.cells.setdefault(key, []).append(idx)
        self.items = items

    def pairs(self, cutoff: float):
        off = int(math.ceil(cutoff / self.cell))
        seen = set()
        for key, idxs in self.cells.items():
            for dx in range(-off, off + 1):
                for dy in range(-off, off + 1):
                    for dz in range(-off, off + 1):
                        other = self.cells.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if not other:
                            continue
                        for i in idxs:
                            for j in other:
                                if i >= j:
                                    continue
                                pair = (i, j)
                                if pair in seen:
                                    continue
                                seen.add(pair)
                                a = self.items[i][1]
                                b = self.items[j][1]
                                d = math.dist(a, b)
                                if d <= cutoff:
                                    yield i, j, d


def distance(a, b) -> float:
    return math.dist(a, b)


def angle_deg(a, b, c) -> float:
    """Angle a-b-c in degrees (b is the vertex)."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    na = math.sqrt(sum(v * v for v in ba))
    nc = math.sqrt(sum(v * v for v in bc))
    if na == 0 or nc == 0:
        return 0.0
    cos = sum(ba[i] * bc[i] for i in range(3)) / (na * nc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


@dataclass
class InteractionRecord:
    target: str
    dataset: str
    seed: str
    sample: str
    method: str
    scope: str
    interaction_class: str
    chain_a: str
    res_a: str
    resname_a: str
    chem_class_a: str
    is_d_a: str
    atom_a: str
    element_a: str
    chain_b: str
    res_b: str
    resname_b: str
    chem_class_b: str
    is_d_b: str
    atom_b: str
    element_b: str
    distance: float
    angle: float
    angle_partner: str
    hydrogen_source: str
    sequence_separation: str
    subset: str
    cutoff_name: str
    cutoff_value: str
    geometry_band: str
    evaluation_status: str = P.EVAL_DETECTED
    reference_id: str = ""
    delta_distance: str = P.NA
    delta_angle: str = P.NA
    notes: str = ""

    def as_row(self, index: int) -> dict:
        row = {"interaction_id": f"{self.target}|{self.method}|{self.seed}|{index:06d}"}
        row.update({k: v for k, v in self.__dict__.items()})
        return row


# ---------------------------------------------------------------------------
#  scope / subset helpers
# ---------------------------------------------------------------------------
def _seqnum(res):
    try:
        return int(str(res.seq).strip())
    except (TypeError, ValueError):
        return None


def _scope(res_a, res_b, atom_a, atom_b) -> str:
    if res_a.is_water or res_b.is_water:
        return P.SCOPE_WATER_MEDIATED
    if (atom_a.element in P.METAL_ELEMENTS) or (atom_b.element in P.METAL_ELEMENTS):
        return P.SCOPE_PROTEIN_METAL
    if res_a.chain != res_b.chain:
        return P.SCOPE_INTERCHAIN
    if res_a.is_polymer and res_b.is_polymer:
        return P.SCOPE_INTRACHAIN
    return P.SCOPE_PROTEIN_COFACTOR


def _subset(res_a, res_b, scope) -> tuple:
    if scope == P.SCOPE_INTRACHAIN:
        ia, ib = _seqnum(res_a), _seqnum(res_b)
        if ia is not None and ib is not None:
            sep = abs(ia - ib)
            return (str(sep), "nonlocal" if sep > P.NONLOCAL_SEPARATION else "local")
        return (P.NA, "unknown")
    return (P.NA, P.NA)


def _disulfide_band(d: float) -> str:
    if P.DISULFIDE_PLAUSIBLE[0] <= d <= P.DISULFIDE_PLAUSIBLE[1]:
        return "plausible_covalent"
    if P.DISULFIDE_STRETCHED[0] < d <= P.DISULFIDE_STRETCHED[1]:
        return "stretched_flagged"
    return "candidate_only"


def _make(meta, cls, scope, res_a, atom_a, res_b, atom_b, d, angle, angle_partner,
          hydrogen_source, cutoff_name, cutoff_value, band, sep, subset, note=""):
    return InteractionRecord(
        target=meta["target"], dataset=meta["dataset"], seed=str(meta["seed"]),
        sample=str(meta["sample"]), method=meta["method"], scope=scope,
        interaction_class=cls, chain_a=res_a.chain, res_a=res_a.token,
        resname_a=res_a.resname, chem_class_a=res_a.chem.chem_class or "",
        is_d_a="yes" if res_a.chem.is_D else "no", atom_a=atom_a.name,
        element_a=atom_a.element, chain_b=res_b.chain, res_b=res_b.token,
        resname_b=res_b.resname, chem_class_b=res_b.chem.chem_class or "",
        is_d_b="yes" if res_b.chem.is_D else "no", atom_b=atom_b.name,
        element_b=atom_b.element, distance=round(d, 3), angle=angle,
        angle_partner=angle_partner, hydrogen_source=hydrogen_source,
        sequence_separation=sep, subset=subset, cutoff_name=cutoff_name,
        cutoff_value=str(cutoff_value), geometry_band=band, notes=note)


def _donor_angle(placed, res, donor_atom, acc_xyz):
    """Best donor-H...acceptor angle for a donor; ``(angle, hydrogen_name)`` or ``(NA, '')``."""
    hs = placed.get((res.chain, res.seq, res.icode, donor_atom.name))
    if not hs:
        return P.NA, ""
    best, name = P.NA, ""
    for hname, hxyz in hs:
        ang = angle_deg(donor_atom.xyz, hxyz, acc_xyz)
        if best == P.NA or ang > best:
            best, name = round(ang, 1), hname
    return best, name


def _hbond_or_polar(meta, placed, res_a, atom_a, roles_a, res_b, atom_b, roles_b, d, hsource,
                    scope, sep, subset):
    """One record for a donor/acceptor pair: ``hbond`` if an H-angle passes, else polar."""
    dirs = []
    if C.ROLE_DONOR in roles_a and C.ROLE_ACCEPTOR in roles_b:
        dirs.append((res_a, atom_a, res_b, atom_b))
    if C.ROLE_DONOR in roles_b and C.ROLE_ACCEPTOR in roles_a:
        dirs.append((res_b, atom_b, res_a, atom_a))
    if not dirs:
        return None
    best = None
    distance_only = None
    for dres, datom, ares, aatom in dirs:
        ang, hname = _donor_angle(placed, dres, datom, aatom.xyz)
        if ang == P.NA:
            if distance_only is None:
                distance_only = (dres, datom, ares, aatom, hname)
            continue
        if ang >= P.HBOND_ANGLE_MIN_DEG and (best is None or ang > best[0]):
            best = (ang, hname, dres, datom, ares, aatom)
    if best is not None:
        ang, hname, dres, datom, ares, aatom = best
        return _make(meta, P.CLASS_HBOND, scope, dres, datom, ares, aatom, d, ang, hname,
                     hsource, "donor_acceptor_distance", P.HBOND_DISTANCE_MAX,
                     f"angle_ge_{int(P.HBOND_ANGLE_MIN_DEG)}", sep, subset)
    if distance_only is not None:
        dres, datom, ares, aatom, hname = distance_only
        return _make(meta, P.CLASS_POLAR_CONTACT, scope, dres, datom, ares, aatom, d, P.NA,
                     "", "unavailable", "donor_acceptor_distance", P.HBOND_DISTANCE_MAX,
                     "distance_only_no_hydrogen", sep, subset,
                     note="donor has no placed hydrogen; not counted as an hbond")
    return None


def detect_all(structure, placed, meta, resolver):
    """Run the frozen detector over one structure and return the interaction records."""
    records = []
    items = []
    for chain in structure.chains:
        for res in chain.residues:
            for a in res.heavy_atoms():
                if res.is_water:
                    roles = {C.ROLE_DONOR, C.ROLE_ACCEPTOR}
                elif a.element in P.METAL_ELEMENTS:
                    roles = set()
                else:
                    roles = resolver.atom_roles(res.resname, a.name)
                    if placed and C.ROLE_DONOR in roles:
                        # a donor with no placed hydrogen (e.g. the unprotonated N of a His
                        # tautomer) is not treated as a donor; polar-contact rows are reserved
                        # for the case where no hydrogen set is available at all.
                        if not placed.get((res.chain, res.seq, res.icode, a.name)):
                            roles = roles - {C.ROLE_DONOR}
                items.append((res, a, roles))
    if not items:
        return records
    grid = Grid([(it, it[1].xyz) for it in items], cell=6.0)
    hsource = "reduce" if placed else "unavailable"
    for i, j, d in grid.pairs(5.2):
        res_a, atom_a, roles_a = items[i]
        res_b, atom_b, roles_b = items[j]
        if res_a is res_b:
            continue
        scope = _scope(res_a, res_b, atom_a, atom_b)
        sep, subset = _subset(res_a, res_b, scope)

        # ---- solvent -----------------------------------------------------
        if res_a.is_water or res_b.is_water:
            if d > P.WATER_BRIDGE_MAX:
                continue
            if res_a.is_water:
                prot_res, prot_atom, prot_roles = res_b, atom_b, roles_b
                wat_res, wat_atom = res_a, atom_a
            else:
                prot_res, prot_atom, prot_roles = res_a, atom_a, roles_a
                wat_res, wat_atom = res_b, atom_b
            if not (prot_roles & {C.ROLE_DONOR, C.ROLE_ACCEPTOR}):
                continue
            records.append(_make(meta, P.CLASS_POLAR_CONTACT, P.SCOPE_WATER_MEDIATED,
                                 prot_res, prot_atom, wat_res, wat_atom, d, P.NA, "",
                                 hsource, "water_bridge", P.WATER_BRIDGE_MAX, "water_contact",
                                 sep, subset, note=f"water {wat_res.token}"))
            continue

        # ---- metal coordination -----------------------------------------
        m_a = atom_a.element in P.METAL_ELEMENTS
        m_b = atom_b.element in P.METAL_ELEMENTS
        if m_a != m_b:
            if m_a:
                metal_res, metal_at, lig_res, lig_at = res_a, atom_a, res_b, atom_b
            else:
                metal_res, metal_at, lig_res, lig_at = res_b, atom_b, res_a, atom_a
            if lig_at.element in ("O", "N", "S"):
                cutoff = (P.covalent_radius(metal_at.element)
                          + P.covalent_radius(lig_at.element)) * P.METAL_RADII_FACTOR
                if d <= cutoff:
                    records.append(_make(meta, P.CLASS_METAL, P.SCOPE_PROTEIN_METAL,
                                         metal_res, metal_at, lig_res, lig_at, d, P.NA, "",
                                         hsource, "metal_ligand_dist_covalent_factor",
                                         round(cutoff, 2), "metal_site", sep, subset))

        # ---- disulfide ---------------------------------------------------
        if (C.ROLE_DISULFIDE in roles_a and C.ROLE_DISULFIDE in roles_b
                and d <= P.DISULFIDE_CANDIDATE_MAX):
            records.append(_make(meta, P.CLASS_DISULFIDE, scope, res_a, atom_a, res_b, atom_b,
                                 d, P.NA, "", hsource, "disulfide_sg_sg",
                                 P.DISULFIDE_CANDIDATE_MAX, _disulfide_band(d), sep, subset))

        # ---- salt bridge -------------------------------------------------
        anion_a, anion_b = C.ROLE_ANION in roles_a, C.ROLE_ANION in roles_b
        cat_a, cat_b = C.ROLE_CATION in roles_a, C.ROLE_CATION in roles_b
        ext_a = C.ROLE_CATION_EXTENDED in roles_a and not cat_a
        ext_b = C.ROLE_CATION_EXTENDED in roles_b and not cat_b
        if d <= P.SALT_BRIDGE_DISTANCE_MAX:
            pair = None
            if anion_a and cat_b:
                pair = (res_a, atom_a, res_b, atom_b, P.CLASS_SALT_BRIDGE)
            elif anion_b and cat_a:
                pair = (res_b, atom_b, res_a, atom_a, P.CLASS_SALT_BRIDGE)
            elif anion_a and ext_b:
                pair = (res_a, atom_a, res_b, atom_b, P.CLASS_SALT_BRIDGE_EXTENDED)
            elif anion_b and ext_a:
                pair = (res_b, atom_b, res_a, atom_a, P.CLASS_SALT_BRIDGE_EXTENDED)
            if pair is not None:
                ra, aa, rb, ab, cls = pair
                records.append(_make(meta, cls, scope, ra, aa, rb, ab, d, P.NA, "", hsource,
                                     "charged_pair_distance", P.SALT_BRIDGE_DISTANCE_MAX,
                                     cls, sep, subset))

        # ---- hydrogen bond / polar contact -------------------------------
        if d <= P.HBOND_DISTANCE_MAX and not (atom_a.element == "S" and atom_b.element == "S"):
            rec = _hbond_or_polar(meta, placed, res_a, atom_a, roles_a, res_b, atom_b, roles_b,
                                  d, hsource, scope, sep, subset)
            if rec is not None:
                records.append(rec)

        # ---- hydrophobic -------------------------------------------------
        if (C.ROLE_HYDROPHOBIC in roles_a and C.ROLE_HYDROPHOBIC in roles_b
                and atom_a.element in P.HYDROPHOBIC_ELEMENTS
                and atom_b.element in P.HYDROPHOBIC_ELEMENTS):
            same_disulfide = (C.ROLE_DISULFIDE in roles_a and C.ROLE_DISULFIDE in roles_b
                              and d <= P.DISULFIDE_CANDIDATE_MAX)
            if not same_disulfide:
                cutoff = (P.vdw_radius(atom_a.element) + P.vdw_radius(atom_b.element)
                          + P.VDW_SLACK)
                if d <= cutoff:
                    records.append(_make(meta, P.CLASS_HYDROPHOBIC, scope, res_a, atom_a,
                                         res_b, atom_b, d, P.NA, "", hsource,
                                         "vdw_sum_plus_slack", round(cutoff, 2),
                                         "hydrophobic_contact", sep, subset))
    return records


# ---------------------------------------------------------------------------
#  comparison against the ground truth (on the evaluable domain)
# ---------------------------------------------------------------------------
def _canon_atom(chem_class: str, atom_name: str) -> str:
    return sorted(M.equivalent_atom_names(chem_class or "", atom_name))[0]


def _atom_pair(rec) -> tuple:
    a = (rec.res_a, _canon_atom(rec.chem_class_a, rec.atom_a))
    b = (rec.res_b, _canon_atom(rec.chem_class_b, rec.atom_b))
    return tuple(sorted([a, b]))


def interaction_key_gt(rec) -> tuple:
    lo, hi = _atom_pair(rec)
    return (rec.interaction_class, rec.scope, lo[0], lo[1], hi[0], hi[1])


def interaction_key_model(rec, forward: dict):
    pa = forward.get((rec.chain_a, rec.res_a))
    pb = forward.get((rec.chain_b, rec.res_b))
    if pa is None or pb is None:
        return None
    a = (pa[1], _canon_atom(rec.chem_class_a, rec.atom_a))
    b = (pb[1], _canon_atom(rec.chem_class_b, rec.atom_b))
    lo, hi = sorted([a, b])
    return (rec.interaction_class, rec.scope, lo[0], lo[1], hi[0], hi[1])


def _cmp_row(meta, rec, gt, status, dd=None, da=None):
    lo, hi = _atom_pair(rec)
    return {
        "target": meta["target"], "dataset": meta["dataset"], "seed": str(meta["seed"]),
        "sample": str(meta["sample"]), "method": meta["method"],
        "interaction_class": rec.interaction_class, "scope": rec.scope, "subset": rec.subset,
        "status": status, "residue_pair": f"{lo[0]}|{hi[0]}", "atom_pair": f"{lo[1]}|{hi[1]}",
        "gt_distance": (gt.distance if gt is not None else P.NA),
        "model_distance": (rec.distance if status != "lost" else P.NA),
        "delta_distance": (P.NA if dd is None else dd),
        "gt_angle": (gt.angle if gt is not None else P.NA),
        "model_angle": (rec.angle if (status == "retained" or status == "retained_but_distorted")
                        else P.NA),
        "delta_angle": (P.NA if da is None else da),
        "provenance": rec.notes,
    }


def compare_to_reference(gt_records, model_records, forward, meta):
    """Classify every model interaction as retained / distorted / extra and GT as lost."""
    domain = {t for (_c, t) in forward.values()}
    gt_map = {}
    for r in gt_records:
        if r.res_a not in domain or r.res_b not in domain:
            continue
        gt_map.setdefault(interaction_key_gt(r), r)
    rows = []
    matched = set()
    for r in model_records:
        key = interaction_key_model(r, forward)
        if key is None:
            rows.append(_cmp_row(meta, r, None, "outside_evaluable_domain"))
            continue
        g = gt_map.get(key)
        if g is None:
            rows.append(_cmp_row(meta, r, None, "extra"))
            continue
        matched.add(key)
        dd = round(r.distance - g.distance, 3)
        da = P.NA
        if r.angle != P.NA and g.angle != P.NA:
            da = round(r.angle - g.angle, 1)
        distorted = abs(dd) > P.RETENTION_DISTANCE_TOLERANCE
        if da != P.NA and abs(da) > P.RETENTION_ANGLE_TOLERANCE:
            distorted = True
        rows.append(_cmp_row(meta, r, g,
                             "retained_but_distorted" if distorted else "retained", dd, da))
    for key, g in gt_map.items():
        if key not in matched:
            rows.append(_cmp_row(meta, g, g, "lost"))
    return rows
