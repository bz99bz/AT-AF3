"""Associative chirality prompt memory for frozen-AF3 conditioning.

The memory is deliberately external to AlphaFold 3. AF3 input JSONs stay in the
standard dialect, while this module builds machine-readable sidecar prompt packages
that describe the chirality constraints AT uses before and after AF3 inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

from utils import D_AMINO_ACID_CCD, ONE_TO_THREE


SIGNED_VOLUME_FORMULA = "dot(cross(N - CA, C - CA), CB - CA)"


@dataclass(frozen=True)
class KnowledgeEntity:
    entity_id: str
    entity_type: str
    case_id: str
    pdb_id: str
    transform_type: str
    chain_scope: str
    residue_scope: str
    expected_chirality: str
    signed_volume_rule: str
    local_geometry_rule: str
    global_fold_rule: str
    excluded_cues: list[str]
    text: str


def _entity_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _case_sequence(case: dict) -> str:
    return case.get("sequence") or case.get("sequence_binder") or ""


def _target_chain_ids(case: dict) -> list[str]:
    ids: list[str] = []
    for target in case.get("target_chains", []):
        chain_ids = target.get("chain_ids")
        if chain_ids:
            ids.extend(chain_ids)
        elif target.get("chain_id"):
            ids.append(target["chain_id"])
    return ids


def _resolved_binder_chain_id(case: dict) -> str:
    """Use the configured binder chain when it is a valid single-letter AF3 id."""
    raw = case.get("binder_chain_id")
    if isinstance(raw, str) and len(raw) == 1 and raw.isalpha() and raw.isupper():
        return raw
    context = case.get("pdb_chain_context", {})
    for key in ("asym_chains", "auth_chains", "chain_ids"):
        values = context.get(key)
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            if isinstance(value, str) and len(value) == 1 and value.isalpha() and value.isupper():
                return value
    return "A"


def _non_gly_positions(sequence: str) -> list[int]:
    return [
        index + 1 for index, aa in enumerate(sequence)
        if ONE_TO_THREE.get(aa) not in {None, "GLY"}
    ]


def resolve_chirality_targets(case: dict) -> dict:
    """Resolve the target chirality scope from case config."""
    transform_type = case.get("transform_type", "")
    sequence = _case_sequence(case)
    binder_chain_id = _resolved_binder_chain_id(case)

    if transform_type == "full_D_chirality_assignment_for_curated_entity":
        binder_case = case.get("case_type") in {
            "d_protein_binder", "d_peptide_binder", "heterochiral_D_binder", "D_peptide_binder",
        }
        if binder_case:
            binder_sequence = case.get("sequence_binder") or sequence
            positions = _non_gly_positions(binder_sequence)
            target_chain_ids = _target_chain_ids(case)
            return {
                "mode": "heterochiral_complex",
                "d_chains": [binder_chain_id],
                "l_chains": target_chain_ids,
                "d_positions": positions,
                "residue_scope": "binder_non_gly",
                "expected_labels": {binder_chain_id: "D", **{cid: "L" for cid in target_chain_ids}},
            }
        positions = _non_gly_positions(sequence)
        return {
            "mode": "all_d_mirror",
            "d_chains": ["A"],
            "l_chains": [],
            "d_positions": positions,
            "residue_scope": "all_non_gly",
            "expected_labels": {"A": "D"},
        }

    if transform_type == "full_mirror_reflection":
        positions = _non_gly_positions(sequence)
        return {
            "mode": "all_d_mirror",
            "d_chains": ["A"],
            "l_chains": [],
            "d_positions": positions,
            "residue_scope": "all_non_gly",
            "expected_labels": {"A": "D"},
        }

    if transform_type == "selective_chain_mirror":
        binder_sequence = case.get("sequence_binder") or sequence
        positions = _non_gly_positions(binder_sequence)
        target_chain_ids = _target_chain_ids(case)
        return {
            "mode": "heterochiral_complex",
            "d_chains": [binder_chain_id],
            "l_chains": target_chain_ids,
            "d_positions": positions,
            "residue_scope": "binder_non_gly",
            "expected_labels": {binder_chain_id: "D", **{cid: "L" for cid in target_chain_ids}},
        }

    if transform_type in ("local_residue_inversion", "local_motif_inversion", "local_chirality_inversion"):
        target_residues = case.get("target_residues", case.get("d_residue_positions", []))
        if target_residues == "last_two":
            target_residues = ["last_two"]
        elif not isinstance(target_residues, list):
            target_residues = []
        return {
            "mode": "site_specific_substitution",
            "d_chains": ["A"],
            "l_chains": [],
            "d_positions": target_residues,
            "residue_scope": "site_specific",
            "expected_labels": {"A": "D_at_target_positions_L_elsewhere"},
        }

    return {
        "mode": "unknown",
        "d_chains": [],
        "l_chains": [],
        "d_positions": [],
        "residue_scope": "unknown",
        "expected_labels": {},
    }


def build_chirality_entities(case: dict) -> list[KnowledgeEntity]:
    targets = resolve_chirality_targets(case)
    coe = case.get("chain_of_entities", {})
    pdb_id = case.get("pdb_id", "unknown")
    case_id = case.get("case_id", "unknown")
    transform_type = case.get("transform_type", "unknown")
    compressed = coe.get("compressed_prompt", case.get("prompt_condition", ""))

    excluded_cues = [
        "natural L-protein prior as a chirality source",
        "unconstrained decorative structural details",
        "residue-name-only D/L labels without coordinate chirality",
    ]
    entities = [
        KnowledgeEntity(
            entity_id=_entity_id(case_id, "chirality", transform_type),
            entity_type="chirality_constraint",
            case_id=case_id,
            pdb_id=pdb_id,
            transform_type=transform_type,
            chain_scope=",".join(targets["d_chains"] + targets["l_chains"]) or "A",
            residue_scope=targets["residue_scope"],
            expected_chirality=json.dumps(targets["expected_labels"], sort_keys=True),
            signed_volume_rule=f"D-like: {SIGNED_VOLUME_FORMULA} < 0; L-like: {SIGNED_VOLUME_FORMULA} > 0; Gly excluded",
            local_geometry_rule="preserve N/CA/C/O and non-Gly CB; allow only target-local relaxation for site-specific D residues",
            global_fold_rule="preserve residue order, chain identity, fold topology, and contact/distance map where constrained",
            excluded_cues=excluded_cues,
            text=compressed,
        ),
        KnowledgeEntity(
            entity_id=_entity_id(case_id, "fold", coe.get("structural_object", "")),
            entity_type="fold_preservation",
            case_id=case_id,
            pdb_id=pdb_id,
            transform_type=transform_type,
            chain_scope="all_modeled_chains",
            residue_scope="global",
            expected_chirality="chain_specific",
            signed_volume_rule="chirality rule is evaluated residue by residue after generation",
            local_geometry_rule=coe.get("expected_structural_effect", "preserve local backbone geometry where constrained"),
            global_fold_rule=coe.get("structural_object", "preserve global fold"),
            excluded_cues=excluded_cues,
            text=coe.get("prompt_conditioned_objective", compressed),
        ),
    ]
    return entities


class AssociativePromptKB:
    """Small in-memory KB using lexical overlap and case metadata for recall."""

    def __init__(self, entities: list[KnowledgeEntity]):
        self.entities = entities

    @classmethod
    def from_cases(cls, cases: list[dict]) -> "AssociativePromptKB":
        entities: list[KnowledgeEntity] = []
        for case in cases:
            entities.extend(build_chirality_entities(case))
        return cls(entities)

    def recall(self, case: dict, top_k: int = 6) -> list[dict]:
        query_terms = {
            str(case.get("case_type", "")).lower(),
            str(case.get("transform_type", "")).lower(),
            str(case.get("pdb_id", "")).lower(),
            resolve_chirality_targets(case)["mode"].lower(),
        }
        scored = []
        for entity in self.entities:
            haystack = " ".join([
                entity.entity_type,
                entity.transform_type,
                entity.pdb_id,
                entity.residue_scope,
                entity.text,
            ]).lower()
            score = sum(1 for term in query_terms if term and term in haystack)
            if entity.transform_type == case.get("transform_type"):
                score += 3
            if entity.case_id == case.get("case_id"):
                score += 4
            if score > 0:
                scored.append({"score": score, "entity": asdict(entity)})
        scored.sort(key=lambda item: (-item["score"], item["entity"]["entity_id"]))
        return scored[:top_k]


def reconstruct_reasoning_path(case: dict, recalled: list[dict]) -> list[str]:
    targets = resolve_chirality_targets(case)
    steps = [
        f"Identify target mode: {targets['mode']} with transform {case.get('transform_type')}.",
        f"Resolve D-chain scope {targets['d_chains']} and L-chain scope {targets['l_chains']}.",
        f"Apply signed-volume convention: {SIGNED_VOLUME_FORMULA}; D < 0, L > 0, Gly excluded.",
        "Preserve only constrained structural cues: D/L state, residue position, local backbone geometry, chain identity, and global fold/contact-map constraints.",
    ]
    if recalled:
        ids = [item["entity"]["entity_id"] for item in recalled]
        steps.append(f"Recall associative KB entities in priority order: {', '.join(ids)}.")
    steps.append("Emit a conditional prompt sidecar while keeping AF3 weights and native AF3 JSON dialect unchanged.")
    return steps


def summarize_chirality_feedback(diagnostics: list[dict]) -> dict:
    failed = []
    checked = 0
    for diag in diagnostics:
        if diag.get("chirality_label") in ("D", "L"):
            checked += 1
        if diag.get("pass_fail") is False:
            failed.append({
                "chain_id": diag.get("chain_id"),
                "residue_id": diag.get("residue_id"),
                "residue_name": diag.get("residue_name"),
                "observed": diag.get("chirality_label"),
                "expected": diag.get("expected_label"),
                "signed_volume": diag.get("chirality_raw_signed_volume"),
            })
    return {
        "checked_residue_count": checked,
        "failed_residue_count": len(failed),
        "failed_residues": failed,
    }


def refine_prompt_with_feedback(prompt_text: str, feedback: Optional[dict]) -> str:
    if not feedback or not feedback.get("failed_residues"):
        return prompt_text
    failed = feedback["failed_residues"]
    failed_positions = ", ".join(
        f"{r.get('chain_id')}:{r.get('residue_id')}({r.get('observed')}->{r.get('expected')})"
        for r in failed[:25]
    )
    return (
        f"{prompt_text}\n\nAT feedback refinement:\n"
        f"Residue-level signed-volume checks failed at {failed_positions}. "
        "Strengthen the conditional prompt for these positions: preserve N/CA/C/O and CB, "
        "force the expected signed-volume orientation, and keep global fold/contact constraints unchanged. "
        "Do not compensate by deleting atoms, changing chain IDs, or globally mirroring unconstrained chains."
    )


def build_conditional_prompt(case: dict, recalled: list[dict], feedback: Optional[dict] = None) -> str:
    targets = resolve_chirality_targets(case)
    coe = case.get("chain_of_entities", {})
    base_prompt = case.get("prompt_condition") or coe.get("compressed_prompt", "")
    recalled_text = " ".join(item["entity"]["text"] for item in recalled[:3] if item["entity"].get("text"))
    prompt = (
        f"AT chirality-conditioned prompt for {case.get('pdb_id')}:\n"
        f"{base_prompt}\n\n"
        f"Resolved target mode: {targets['mode']}.\n"
        f"D chains: {targets['d_chains']}; L chains: {targets['l_chains']}; D positions: {targets['d_positions']}.\n"
        f"Signed-volume rule: D-like {SIGNED_VOLUME_FORMULA} < 0, L-like {SIGNED_VOLUME_FORMULA} > 0; Gly excluded.\n"
        "Relevant structural cues only: D/L state, residue position, signed-volume orientation, local backbone geometry, chain identity, and global fold preservation.\n"
        "Exclude unconstrained cues and do not rely on residue labels alone for chirality.\n"
        f"Associative recall context: {recalled_text}"
    )
    return refine_prompt_with_feedback(prompt, feedback)


def build_prompt_package(
    case: dict,
    cases: Optional[list[dict]] = None,
    feedback: Optional[dict] = None,
    leave_one_case_out: bool = False,
) -> dict:
    kb_cases = cases or [case]
    if leave_one_case_out:
        query_case_id = case.get("case_id")
        query_pdb_id = case.get("pdb_id")
        kb_cases = [
            kb_case for kb_case in kb_cases
            if kb_case.get("case_id") != query_case_id and kb_case.get("pdb_id") != query_pdb_id
        ]
    kb = AssociativePromptKB.from_cases(kb_cases)
    recalled = kb.recall(case)
    prompt_text = build_conditional_prompt(case, recalled, feedback)
    return {
        "case_id": case.get("case_id"),
        "pdb_id": case.get("pdb_id"),
        "af3_weights_status": "FROZEN",
        "af3_json_policy": "standard_alphafold3_json_plus_external_at_prompt_sidecar",
        "chirality_targets": resolve_chirality_targets(case),
        "knowledge_entities": [asdict(e) for e in build_chirality_entities(case)],
        "recalled_entities": recalled,
        "reasoning_path": reconstruct_reasoning_path(case, recalled),
        "conditional_prompt": prompt_text,
        "feedback": feedback or {"checked_residue_count": 0, "failed_residue_count": 0, "failed_residues": []},
        "d_amino_acid_ccd_map": D_AMINO_ACID_CCD,
    }

