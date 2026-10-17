"""Generate AlphaFold 3 JSON inputs for the CHORD generation conditions.

Standard-AF3 inputs contain the unmodified L-amino-acid sequence. AT-AF3 and
Oracle-CCD inputs additionally attach D-amino-acid CCD components as AF3
modifications; the residue-level target positions and the associative prompt
package are recorded separately so the input remains a standard AF3 JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from utils import (
    load_cases,
    ensure_output_dirs,
    D_AMINO_ACID_CCD,
    ONE_TO_THREE,
    PROJECT_ROOT,
)
from associative_chirality_memory import build_prompt_package

logger = logging.getLogger(__name__)

AF3_INPUTS_DIR = PROJECT_ROOT / "af3_inputs"
BASELINE_INPUTS_DIR = AF3_INPUTS_DIR / "baseline"
PROMPT_INPUTS_DIR = AF3_INPUTS_DIR / "prompt_conditioned"
ORACLE_INPUTS_DIR = AF3_INPUTS_DIR / "oracle_ccd"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

ALL_D_CASE_TYPES = {
    "all_D_mirror_image",
    "all_D_racemic_crystallography",
    "all_D_de_novo_or_mirror_image",
}
BINDER_CASE_TYPES = {
    "d_protein_binder",
    "d_peptide_binder",
    "heterochiral_D_binder",
    "D_peptide_binder",
}
LOCAL_D_CASE_TYPES = {"local_d_residue", "d_asp_racemization"}


def _allow_unknown(case: dict) -> bool:
    return bool(case.get("allow_unknown", False)) or "X" in case.get("sequence", "") or "X" in case.get("sequence_binder", "")


def _primary_sequence(case: dict) -> str:
    return case.get("sequence_binder") or case.get("sequence", "")


def _target_chain_ids(case: dict) -> list[str]:
    ids: list[str] = []
    for target in case.get("target_chains", []):
        chain_ids = target.get("chain_ids")
        if chain_ids:
            ids.extend(chain_ids)
        elif target.get("chain_id"):
            ids.append(target["chain_id"])
    return ids


def _is_valid_af3_chain_id(chain_id: object) -> bool:
    return isinstance(chain_id, str) and len(chain_id) == 1 and chain_id.isalpha() and chain_id.isupper()


def _next_available_chain_id(used_ids: set[str]) -> str:
    for code in range(ord("A"), ord("Z") + 1):
        chain_id = chr(code)
        if chain_id not in used_ids:
            return chain_id
    raise ValueError("Too many chains: AF3 input ids are limited to A-Z")


def _first_valid_chain_id(values: object, used_ids: set[str] | None = None) -> str | None:
    used_ids = used_ids or set()
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, list):
        candidates = [str(v) for v in values]
    else:
        candidates = []
    for candidate in candidates:
        if _is_valid_af3_chain_id(candidate) and candidate not in used_ids:
            return candidate
    return None


def _binder_chain_id(case: dict, used_ids: set[str] | None = None) -> str:
    used_ids = used_ids or set()
    explicit = _first_valid_chain_id(case.get("binder_chain_id"), used_ids)
    if explicit:
        return explicit
    context = case.get("pdb_chain_context", {})
    for key in ("asym_chains", "auth_chains", "chain_ids"):
        candidate = _first_valid_chain_id(context.get(key), used_ids)
        if candidate:
            return candidate
    return _next_available_chain_id(used_ids)


def _chain_id_from_target(target: dict, used_ids: set[str]) -> str:
    for key in ("chain_id", "chain_ids", "asym_chains", "auth_chains"):
        candidate = _first_valid_chain_id(target.get(key), used_ids)
        if candidate:
            return candidate
    return _next_available_chain_id(used_ids)


def validate_protein_sequence(sequence: str, label: str, allow_unknown: bool = False):
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"{label}: sequence must be a non-empty string")
    allowed = set(VALID_AA)
    if allow_unknown:
        allowed.add("X")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise ValueError(
            f"{label}: invalid amino-acid sequence characters {invalid}; "
            "only ACDEFGHIKLMNPQRSTVWY are allowed unless allow_unknown is true"
        )


def deterministic_seed(case: dict, seed: int | None = None) -> int:
    if seed is not None:
        if seed <= 0:
            raise ValueError("--seed must be a positive integer")
        return seed
    key = f"{case['case_id']}:{case['pdb_id']}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1) + 1


def _make_protein_chain(
    chain_id: str,
    sequence: str,
    modifications: list[dict] | None = None,
) -> dict:
    chain = {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "modifications": modifications or [],
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": [],
        }
    }
    return chain



def _all_d_chain_sequences(case: dict) -> list[tuple[str, str]]:
    """Return chain-specific sequences for all-D mirror cases."""
    allow_unknown = _allow_unknown(case)
    chain_sequences = case.get("chain_sequences")
    if isinstance(chain_sequences, dict) and chain_sequences:
        chains = [(str(chain_id), str(seq)) for chain_id, seq in chain_sequences.items()]
    else:
        sequence = case["sequence"]
        parts = sequence.split("|") if "|" in sequence else [sequence]
        chains = [(chr(ord("A") + i), seq) for i, seq in enumerate(parts)]

    seen_ids = set()
    for chain_id, sequence in chains:
        if chain_id in seen_ids:
            raise ValueError(f"{case['case_id']} {case['pdb_id']}: duplicate chain id {chain_id}")
        validate_protein_sequence(
            sequence,
            f"{case['case_id']} {case['pdb_id']} chain {chain_id}",
            allow_unknown=allow_unknown,
        )
        seen_ids.add(chain_id)
    return chains


def _all_d_modifications_for_sequence(sequence: str) -> list[dict]:
    modifications = []
    for i, aa in enumerate(sequence):
        three_letter = ONE_TO_THREE.get(aa)
        if three_letter and three_letter in D_AMINO_ACID_CCD and three_letter != "GLY":
            modifications.append({
                "ptmType": D_AMINO_ACID_CCD[three_letter],
                "ptmPosition": i + 1,
            })
    return modifications
def _clean_generated_inputs(output_base: Path):
    """Remove stale generated AF3 inputs before writing a fresh set."""
    for subdir in (
        output_base / "baseline",
        output_base / "prompt_conditioned",
        output_base / "oracle_ccd",
        output_base / "at_prompt_sidecars",
    ):
        if subdir.exists():
            for child in subdir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            subdir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_base / "af3_inputs_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _binder_target_chains(case: dict, binder_modifications: list[dict] | None = None) -> list[dict]:
    """Build binder + receptor chains for heterochiral binder cases.

    Binder cases must be multi-chain inputs. The optimized config carries
    receptor sequences in target_chains; descriptive sequence_target text is not
    enough to make AF3 produce an interface.
    """
    binder_seq = _primary_sequence(case)
    target_chains = case.get("target_chains", [])
    binder_chain_id = _binder_chain_id(case)
    allow_unknown = _allow_unknown(case)
    if not binder_seq:
        raise ValueError(f"{case['case_id']} {case['pdb_id']}: missing sequence_binder")
    validate_protein_sequence(
        binder_seq,
        f"{case['case_id']} {case['pdb_id']} sequence_binder",
        allow_unknown=allow_unknown,
    )
    if not target_chains:
        raise ValueError(
            f"{case['case_id']} {case['pdb_id']}: binder cases require target_chains "
            "so AF3 input contains binder + receptor chain(s)"
        )

    chains = [_make_protein_chain(binder_chain_id, binder_seq, binder_modifications or [])]
    seen_ids = {binder_chain_id}
    for target in target_chains:
        chain_id = _chain_id_from_target(target, seen_ids)
        sequence = target.get("sequence")
        if not chain_id or not sequence:
            raise ValueError(
                f"{case['case_id']} {case['pdb_id']}: each target_chains entry "
                "must include chain_id and sequence"
            )
        if chain_id in seen_ids:
            raise ValueError(f"{case['case_id']} {case['pdb_id']}: duplicate chain id {chain_id}")
        validate_protein_sequence(
            sequence,
            f"{case['case_id']} {case['pdb_id']} target chain {chain_id}",
            allow_unknown=allow_unknown or bool(target.get("allow_unknown", False)),
        )
        seen_ids.add(chain_id)
        chains.append(_make_protein_chain(chain_id, sequence))
    return chains


def generate_baseline_input(case: dict, seed: int | None = None) -> dict:
    """Generate a standard AF3 baseline input (no D-amino acid info)."""
    pdb_id = case["pdb_id"]
    case_type = case["case_type"]

    if case_type in ALL_D_CASE_TYPES:
        chains = [
            _make_protein_chain(chain_id, sequence)
            for chain_id, sequence in _all_d_chain_sequences(case)
        ]
    elif case_type in BINDER_CASE_TYPES:
        chains = _binder_target_chains(case)
    elif case_type in LOCAL_D_CASE_TYPES:
        sequence = case["sequence"]
        validate_protein_sequence(sequence, f"{case['case_id']} {pdb_id} sequence", allow_unknown=_allow_unknown(case))
        chains = [_make_protein_chain("A", sequence)]
    elif case_type == "d_ala_d_ser_motif":
        sequence = case.get("sequence", "AAAAAAS")
        chains = [_make_protein_chain("A", sequence)]
    else:
        sequence = case.get("sequence", "AAAA")
        validate_protein_sequence(sequence, f"{case['case_id']} {pdb_id} sequence", allow_unknown=_allow_unknown(case))
        chains = [_make_protein_chain("A", sequence)]

    seed = deterministic_seed(case, seed)
    af3_input = {
        "name": f"{pdb_id}_baseline",
        "modelSeeds": [seed],
        "sequences": chains,
        "dialect": "alphafold3",
        "version": 2,
    }
    return af3_input


def generate_prompt_conditioned_input(case: dict, seed: int | None = None) -> dict:
    """Generate an AF3 input with D-amino acid CCD modifications where possible.

    This is the 'prompt-conditioned' input. The actual stereochemistry
    correction is applied post-prediction, but we include CCD modifications
    as informational markers.
    """
    pdb_id = case["pdb_id"]
    case_type = case["case_type"]

    if case_type in ALL_D_CASE_TYPES:
        chains = [
            _make_protein_chain(chain_id, sequence, _all_d_modifications_for_sequence(sequence))
            for chain_id, sequence in _all_d_chain_sequences(case)
        ]

    elif case_type in BINDER_CASE_TYPES:
        binder_seq = _primary_sequence(case)
        validate_protein_sequence(
            binder_seq,
            f"{case['case_id']} {pdb_id} sequence_binder",
            allow_unknown=_allow_unknown(case),
        )
        modifications = []
        for i, aa in enumerate(binder_seq):
            three_letter = ONE_TO_THREE.get(aa)
            if three_letter and three_letter in D_AMINO_ACID_CCD and three_letter != "GLY":
                modifications.append({
                    "ptmType": D_AMINO_ACID_CCD[three_letter],
                    "ptmPosition": i + 1,
                })
        chains = _binder_target_chains(case, modifications)

    elif case_type in LOCAL_D_CASE_TYPES:
        sequence = case["sequence"]
        validate_protein_sequence(sequence, f"{case['case_id']} {pdb_id} sequence", allow_unknown=_allow_unknown(case))
        d_positions = case.get("d_residue_positions", [])
        d_types = case.get("d_ccd_codes", [])
        modifications = []
        if isinstance(d_positions, list):
            for i, pos in enumerate(d_positions):
                ccd_code = d_types[i] if i < len(d_types) else d_types[0]
                modifications.append({
                    "ptmType": ccd_code,
                    "ptmPosition": pos,
                })
        chains = [_make_protein_chain("A", sequence, modifications)]

    elif case_type == "d_ala_d_ser_motif":
        sequence = case.get("sequence", "AAAAAAS")
        modifications = [
            {"ptmType": "DAL", "ptmPosition": len(sequence) - 1},
            {"ptmType": "DSN", "ptmPosition": len(sequence)},
        ]
        chains = [_make_protein_chain("A", sequence, modifications)]

    else:
        sequence = case.get("sequence", "AAAA")
        validate_protein_sequence(sequence, f"{case['case_id']} {pdb_id} sequence", allow_unknown=_allow_unknown(case))
        chains = [_make_protein_chain("A", sequence)]

    seed = deterministic_seed(case, seed)
    af3_input = {
        "name": f"{pdb_id}_prompt_D",
        "modelSeeds": [seed],
        "sequences": chains,
        "dialect": "alphafold3",
        "version": 2,
    }
    return af3_input


def generate_oracle_ccd_input(case: dict, seed: int | None = None) -> dict:
    """Generate strict AF3 + Oracle CCD input.

    This control receives only the coarse chirality class: all-D or local-D.
    It must not use AT retrieval, prompt reasoning, residue-level explanations,
    or inferred local-D target positions. For all-D cases, the coarse label is
    enough to mechanically mark every non-Gly residue as D. For local-D cases,
    the coarse label alone does not identify a residue, so no site-specific
    D-CCD modification is emitted.
    """
    pdb_id = case["pdb_id"]
    case_type = case["case_type"]

    if case_type in ALL_D_CASE_TYPES:
        chains = [
            _make_protein_chain(chain_id, sequence, _all_d_modifications_for_sequence(sequence))
            for chain_id, sequence in _all_d_chain_sequences(case)
        ]
    elif case_type in BINDER_CASE_TYPES:
        chains = _binder_target_chains(
            case, _all_d_modifications_for_sequence(_primary_sequence(case))
        )
    else:
        baseline = generate_baseline_input(case, seed=seed)
        chains = baseline["sequences"]

    return {
        "name": f"{pdb_id}_oracle_D",
        "modelSeeds": [deterministic_seed(case, seed)],
        "sequences": chains,
        "dialect": "alphafold3",
        "version": 2,
    }


def generate_all_inputs(
    cases: list[dict],
    output_base: Path | None = None,
    clean_output: bool = True,
    leave_one_case_out: bool = False,
    seed: int | None = None,
):
    """Generate all AF3 input JSON files."""
    if output_base is None:
        output_base = AF3_INPUTS_DIR

    baseline_dir = output_base / "baseline"
    prompt_dir = output_base / "prompt_conditioned"
    oracle_dir = output_base / "oracle_ccd"
    sidecar_dir = output_base / "at_prompt_sidecars"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        _clean_generated_inputs(output_base)

    manifest = []
    for case in cases:
        pdb_id = case["pdb_id"]

        baseline = generate_baseline_input(case, seed=seed)
        baseline_path = baseline_dir / f"{pdb_id}_baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Written baseline input: {baseline_path}")

        oracle = generate_oracle_ccd_input(case, seed=seed)
        oracle_path = oracle_dir / f"{pdb_id}_oracle_D.json"
        with open(oracle_path, "w") as f:
            json.dump(oracle, f, indent=2)
        logger.info(f"Written oracle CCD input: {oracle_path}")

        prompt = generate_prompt_conditioned_input(case, seed=seed)
        prompt_path = prompt_dir / f"{pdb_id}_prompt_D.json"
        with open(prompt_path, "w") as f:
            json.dump(prompt, f, indent=2)
        logger.info(f"Written prompt-conditioned input: {prompt_path}")

        prompt_package = build_prompt_package(case, cases, leave_one_case_out=leave_one_case_out)
        sidecar_path = sidecar_dir / f"{pdb_id}_at_prompt.json"
        with open(sidecar_path, "w") as f:
            json.dump(prompt_package, f, indent=2)
        logger.info(f"Written AT prompt sidecar: {sidecar_path}")

        baseline_chains = [entry["protein"]["id"] for entry in baseline["sequences"]]
        prompt_chains = [entry["protein"]["id"] for entry in prompt["sequences"]]
        prompt_modified_chains = [
            entry["protein"]["id"]
            for entry in prompt["sequences"]
            if entry["protein"].get("modifications")
        ]
        sequence_lengths = {
            entry["protein"]["id"]: len(entry["protein"]["sequence"])
            for entry in prompt["sequences"]
        }
        manifest.append({
            "case_id": case["case_id"],
            "pdb_id": pdb_id,
            "case_type": case["case_type"],
            "baseline_chain_ids": baseline_chains,
            "prompt_chain_ids": prompt_chains,
            "binder_chain_id": _binder_chain_id(case) if case["case_type"] in BINDER_CASE_TYPES else None,
            "target_chain_ids": _target_chain_ids(case),
            "prompt_modified_chains": prompt_modified_chains,
            "sequence_lengths": sequence_lengths,
            "modelSeeds": baseline["modelSeeds"],
            "oracle_input": str(oracle_path),
            "prompt_input": str(prompt_path),
            "at_prompt_sidecar": str(sidecar_path),
            "retrieval_policy": "leave_one_case_out" if leave_one_case_out else "full_config",
            "at_recalled_entity_ids": [
                item["entity"]["entity_id"]
                for item in prompt_package["recalled_entities"]
            ],
            "at_reasoning_path": prompt_package["reasoning_path"],
        })

    manifest_path = output_base / "af3_inputs_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Written AF3 inputs manifest: {manifest_path}")


def parse_seed_args(args) -> list[int]:
    """Return requested AF3 seeds, preserving explicit user-provided lists."""
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    elif args.seed_start is not None or args.num_seeds is not None:
        seed_start = args.seed_start if args.seed_start is not None else 101
        num_seeds = args.num_seeds if args.num_seeds is not None else 10
        if num_seeds <= 0:
            raise ValueError("--num-seeds must be positive")
        seeds = [seed_start + i for i in range(num_seeds)]
    else:
        seeds = [101]

    if any(seed <= 0 for seed in seeds):
        raise ValueError("AF3 seeds must be positive integers")
    if len(seeds) == 1:
        logger.warning(
            "Only one AF3 seed was evaluated. Seed-level uncertainty and error bars cannot be estimated."
        )
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Generate AlphaFold 3 input JSON files.")
    parser.add_argument("--config", default=None, help="Path to cases JSON config.")
    parser.add_argument("--out", default=None, help="Output root for af3_inputs.")
    parser.add_argument("--loco-retrieval", action="store_true", help="Build AT sidecars with leave-one-case-out retrieval.")
    parser.add_argument("--seed", type=int, default=None, help="Use one explicit AF3 model seed for every generated input.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Generate one input tree per AF3 seed.")
    parser.add_argument("--seed-start", type=int, default=None, help="First seed for generated seed range, used with --num-seeds.")
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of seeds to generate with --seed-start.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cases = load_cases(args.config)
    seeds = parse_seed_args(args)
    output_base = Path(args.out) if args.out else AF3_INPUTS_DIR

    if len(seeds) == 1 and not args.seeds and args.seed_start is None and args.num_seeds is None:
        generate_all_inputs(cases, output_base=output_base, leave_one_case_out=args.loco_retrieval, seed=seeds[0])
        print(f"Generated {len(cases) * 3} AF3 input files")
    else:
        for seed in seeds:
            seed_out = output_base / f"seed_{seed}"
            generate_all_inputs(cases, output_base=seed_out, leave_one_case_out=args.loco_retrieval, seed=seed)
        manifest = {
            "config": args.config,
            "seeds": seeds,
            "seed_input_dirs": {str(seed): str(output_base / f"seed_{seed}") for seed in seeds},
        }
        output_base.mkdir(parents=True, exist_ok=True)
        with open(output_base / "multiseed_inputs_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Generated {len(cases) * 3 * len(seeds)} AF3 input files across {len(seeds)} seeds")


if __name__ == "__main__":
    main()




