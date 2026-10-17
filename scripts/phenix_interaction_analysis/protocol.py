"""Frozen protocol for the Phenix-based interaction analysis (GT / AF3 / AT-AF3).

Every numeric threshold that is *not* a published rule is listed in
:data:`PROTOCOL_CHOICES` with its label and rationale.  The values are frozen and applied
**identically to every method** (GT, AF3, uncorrected AT-AF3, corrected AT-AF3) so that a
difference between methods is a difference in the structures, never in the detector.

The five primary interaction classes, the scope split used here
(intrachain / interchain / protein-metal / water-mediated) and the evaluable-domain
bookkeeping live here.  Nothing in this package minimises, rebuilds, re-flips or repairs a
structure: only hydrogens are added (Phenix ``reduce``) and the heavy atoms are checked to
be unchanged.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  method / condition labels  (canonical file-name suffixes live in manifest.py)
# ---------------------------------------------------------------------------
METHOD_GT = "GT"
METHOD_AF3 = "AF3"
METHOD_ATAF3 = "AT-AF3"              # uncorrected AT-AF3 (D-prompt model)
METHOD_ATAF3_CORR = "Corrected AT-AF3"
METHODS = (METHOD_GT, METHOD_AF3, METHOD_ATAF3, METHOD_ATAF3_CORR)
MODEL_METHODS = (METHOD_AF3, METHOD_ATAF3, METHOD_ATAF3_CORR)
# canonical suffix <-> method label
SUFFIX_TO_METHOD = {"GT": METHOD_GT, "AF3": METHOD_AF3, "AT_AF3": METHOD_ATAF3,
                    "Corrected_AT_AF3": METHOD_ATAF3_CORR}
METHOD_TO_SUFFIX = {v: k for k, v in SUFFIX_TO_METHOD.items()}

# ---------------------------------------------------------------------------
#  interaction classes and scopes
# ---------------------------------------------------------------------------
CLASS_DISULFIDE = "disulfide"
CLASS_SALT_BRIDGE = "salt_bridge"
CLASS_SALT_BRIDGE_EXTENDED = "salt_bridge_extended"
CLASS_METAL = "metal_coordination"
CLASS_HBOND = "hbond"
CLASS_POLAR_CONTACT = "polar_contact_heavy_atom"
CLASS_HYDROPHOBIC = "hydrophobic"
CLASS_VDW = "vdw_contact"
CLASS_CLASH = "steric_overlap"
PRIMARY_CLASSES = (CLASS_DISULFIDE, CLASS_SALT_BRIDGE, CLASS_METAL, CLASS_HBOND, CLASS_HYDROPHOBIC)
AUX_CLASSES = (CLASS_SALT_BRIDGE_EXTENDED, CLASS_POLAR_CONTACT, CLASS_VDW, CLASS_CLASH)

SCOPE_INTRACHAIN = "intrachain"
SCOPE_INTERCHAIN = "interchain"
SCOPE_PROTEIN_METAL = "protein_metal"
SCOPE_WATER_MEDIATED = "water_mediated"
SCOPE_PROTEIN_COFACTOR = "protein_cofactor"
SCOPE_OTHER = "other_or_unknown"
SCOPES = (SCOPE_INTRACHAIN, SCOPE_INTERCHAIN, SCOPE_PROTEIN_METAL, SCOPE_WATER_MEDIATED,
          SCOPE_PROTEIN_COFACTOR, SCOPE_OTHER)

# ---------------------------------------------------------------------------
#  evaluation statuses (one per interaction record)
# ---------------------------------------------------------------------------
EVAL_EVALUABLE = "evaluable"
EVAL_UNSUPPORTED_CHEMISTRY = "unsupported_chemistry"
EVAL_MISSING_ATOMS = "missing_atoms"
EVAL_OUT_OF_DOMAIN = "outside_evaluable_domain"
EVAL_REFERENCE_ABSENT = "reference_absent"
EVAL_DETECTED = "detected"
EVAL_REFERENCE = "reference"          # status carried by every ground-truth record
NA = "NA"

# ---------------------------------------------------------------------------
#  summary reference bookkeeping (see run._summarise)
# ---------------------------------------------------------------------------
#: the reference interaction was compared against the model (even if every record was lost)
REFERENCE_COMPARED = "compared"
#: the reference class/scope exists in GT but the model has no such interaction at all
#: (e.g. protein-metal / water-mediated: the model simply does not contain the ion/water)
REFERENCE_MODEL_ABSENT = "reference_only_model_absent"
#: the model produced interactions of this class/scope although GT has none (pure extra)
REFERENCE_NONE = "model_only_no_reference"

# ---------------------------------------------------------------------------
#  thresholds (frozen; applied to every method)
# ---------------------------------------------------------------------------
HBOND_DISTANCE_MAX = 3.5          # donor-acceptor heavy atom distance (GetContacts)
HBOND_ANGLE_MIN_DEG = 120.0       # D-H...A angle (protocol choice)
HBOND_H_LENGTH = 1.3              # D-H bond length used to identify the attached hydrogen
SALT_BRIDGE_DISTANCE_MAX = 4.0    # charged-atom distance (GetContacts)
DISULFIDE_CANDIDATE_MAX = 2.5     # SG-SG candidate
DISULFIDE_PLAUSIBLE = (1.90, 2.20)
DISULFIDE_STRETCHED = (2.20, 2.50)
VDW_SLACK = 0.5                   # r_vdw(A)+r_vdw(B)+slack (GetContacts)
STERIC_OVERLAP_MIN = 0.4          # overlap = r_a+r_b-d > this
METAL_RADII_FACTOR = 1.30         # r_cov(metal)+r_cov(ligand) scaled
WATER_BRIDGE_MAX = 3.5            # protein atom ... water O distance (same as hbond)
HYDROPHOBIC_RESIDUES = ("ALA", "CYS", "PHE", "GLY", "ILE", "LEU", "MET", "PRO", "VAL", "TRP")
HYDROPHOBIC_ELEMENTS = ("C", "S")
NONLOCAL_SEPARATION = 4           # |i-j| > 4 along the mapped polymer
RETENTION_DISTANCE_TOLERANCE = 1.0    # retained-but-distorted band (angstrom)
RETENTION_ANGLE_TOLERANCE = 15.0      # retained-but-distorted band (degrees)

PROTONATION_POLICY = "reference-state-pH7 (ASP/GLU charged, LYS/ARG charged, HIS neutral)"
ALTLOC_POLICY = "keep highest-occupancy altloc per atom group (tie: A,B,C,...); drop the rest"

METAL_ELEMENTS = frozenset({
    "LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "AL", "GA", "IN", "TL", "PB", "BI",
})
WATER_NAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "SOL"})
ORGANIC_ELEMENTS = frozenset({"C", "N", "O", "S", "P", "H", "SE", "F", "CL", "BR", "I", "B"})

VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "B": 1.92,
    "NA": 2.27, "K": 2.75, "MG": 1.73, "CA": 2.31, "MN": 2.05, "FE": 2.05,
    "CO": 2.00, "NI": 1.97, "CU": 1.96, "ZN": 2.01, "CD": 2.18, "HG": 2.06,
}
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "SE": 1.20,
    "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "NA": 1.66, "K": 2.03, "MG": 1.41, "CA": 1.76, "MN": 1.39, "FE": 1.32,
    "CO": 1.26, "NI": 1.24, "CU": 1.32, "ZN": 1.22, "CD": 1.44, "HG": 1.32,
    "LI": 1.28, "RB": 2.20, "CS": 2.44, "SR": 1.95, "BA": 2.15, "MO": 1.54, "W": 1.62,
}


def vdw_radius(element: str) -> float:
    return VDW_RADII.get((element or "").upper(), 1.70)


def covalent_radius(element: str) -> float:
    return COVALENT_RADII.get((element or "").upper(), 0.76)


# ---------------------------------------------------------------------------
#  standard amino-acid chemistry (parent, L) - used for atom typing
# ---------------------------------------------------------------------------
STANDARD_AA = ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
               "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")

# side-chain hydrogen-bond donors / acceptors (heavy atoms); backbone N/O are added for all.
SIDE_CHAIN_DONORS = {
    "ARG": ("NE", "NH1", "NH2"), "ASN": ("ND2",), "CYS": ("SG",), "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"), "LYS": ("NZ",), "SER": ("OG",), "THR": ("OG1",),
    "TRP": ("NE1",), "TYR": ("OH",),
}
SIDE_CHAIN_ACCEPTORS = {
    "ASN": ("OD1",), "ASP": ("OD1", "OD2"), "CYS": ("SG",), "GLN": ("OE1",),
    "GLU": ("OE1", "OE2"), "HIS": ("ND1", "NE2"), "SER": ("OG",), "THR": ("OG1",),
    "TYR": ("OH",),
}
SIDE_CHAIN_CATIONS = {"LYS": ("NZ",), "ARG": ("NH1", "NH2")}
SIDE_CHAIN_ANIONS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
# extended (never merged into the primary class): protonated His
SIDE_CHAIN_CATIONS_EXTENDED = {"HIS": ("ND1", "NE2"), "LYS": ("NZ",), "ARG": ("NH1", "NH2")}

# chemically interchangeable atom names by residue (symmetry orbits of the CCD graph),
# used when matching an atom-level contact between GT and a model.
EQUIVALENT_ATOMS = {
    "ASP": (("OD1", "OD2"),), "GLU": (("OE1", "OE2"),), "ARG": (("NH1", "NH2"),),
    "LEU": (("CD1", "CD2"),), "VAL": (("CG1", "CG2"),), "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
}

# ---------------------------------------------------------------------------
#  published / protocol provenance
# ---------------------------------------------------------------------------
PROTOCOL_ID = "phenix-interaction-v1"
PROTOCOL_CHOICES = {
    "hbond_angle": {"value": HBOND_ANGLE_MIN_DEG, "unit": "degrees",
                    "label": "protocol choice",
                    "why": "donor-H...acceptor angle; hydrogens come from Phenix reduce and "
                           "the provenance is recorded per structure."},
    "hbond_distance": {"value": HBOND_DISTANCE_MAX, "unit": "angstrom",
                       "label": "GetContacts documented rule"},
    "salt_bridge_distance": {"value": SALT_BRIDGE_DISTANCE_MAX, "unit": "angstrom",
                             "label": "GetContacts documented rule"},
    "disulfide_band": {"value": {"candidate": DISULFIDE_CANDIDATE_MAX,
                                 "plausible": list(DISULFIDE_PLAUSIBLE),
                                 "stretched": list(DISULFIDE_STRETCHED)},
                       "unit": "angstrom", "label": "protocol choice"},
    "hydrophobic_vdw_slack": {"value": VDW_SLACK, "unit": "angstrom",
                              "label": "GetContacts documented rule"},
    "metal_cutoff": {"value": METAL_RADII_FACTOR,
                     "unit": "multiple of summed covalent radii", "label": "protocol choice"},
    "water_bridge": {"value": WATER_BRIDGE_MAX, "unit": "angstrom", "label": "protocol choice"},
    "retention_band": {"value": {"distance": RETENTION_DISTANCE_TOLERANCE,
                                 "angle": RETENTION_ANGLE_TOLERANCE},
                       "unit": "angstrom / degrees", "label": "protocol choice"},
    "protonation": {"value": PROTONATION_POLICY, "label": "protocol choice"},
    "altloc": {"value": ALTLOC_POLICY, "label": "protocol choice"},
}

PROVENANCE_NOTES = (
    "Interaction geometry alone does not establish thermodynamic stability.  Distance and "
    "unsigned-angle contacts are invariant to a complete reflection of all coordinates, so a "
    "signed-volume chirality assessment is required separately.  An extra predicted contact is "
    "absent from the chosen reference, which does not by itself make it chemically impossible."
)
