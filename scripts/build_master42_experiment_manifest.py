"""Materialize the frozen master42 AF3/AT-AF3 generation task manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_af3_inputs import generate_baseline_input, generate_prompt_conditioned_input
from utils import load_cases

EXPECTED_CASES = {"7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"}
FIELDS = [
    "case_id", "pdb_entity_id", "pdb_id", "seed_index", "inference_seed", "condition",
    "input_json", "output_dir", "target_config", "prompt_enabled", "all_D_condition",
    "D_CCD_enabled", "model_version", "code_commit", "config_sha256", "status",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not_available_no_git_repository"


def entity_id(case: dict) -> str:
    return f"{case['pdb_id']}_{case.get('entity_id', case['case_id'].rsplit('_', 1)[-1])}"


def target_config_for_case(case: dict) -> dict:
    is_binder = case["case_type"] in {"d_protein_binder", "d_peptide_binder", "heterochiral_D_binder", "D_peptide_binder"}
    if is_binder:
        return {
            "case_id": entity_id(case), "mode": "selective_D_binder",
            "chains": {
                "A": {"default_target_chirality": "D", "exclude_residue_names": ["GLY"]},
                "B": {"default_target_chirality": "L", "allow_coordinate_modification": False},
            },
            "reference_coordinates_allowed": False,
        }
    return {
        "case_id": entity_id(case), "mode": "all_D",
        "chains": {"A": {"default_target_chirality": "D", "exclude_residue_names": ["GLY"]}},
        "reference_coordinates_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fresh master42 inference inputs and n-seed task manifest.")
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--case-config", default="configs/at_af_7.json")
    parser.add_argument("--case-config-root", default=None, help="Compatibility argument; the frozen case config is authoritative.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", default="runs_master42_n30")
    parser.add_argument("--model-version", default="alphafold3_frozen")
    args = parser.parse_args(argv)

    seed_payload = json.loads(Path(args.seed_file).read_text(encoding="utf-8"))
    seeds = [int(value) for value in seed_payload["inference_seeds"]]
    if seed_payload.get("master_seed") != 42 or len(seeds) != 30:
        raise ValueError("master42 n30 experiment requires exactly 30 seeds generated from master_seed=42")
    cases = load_cases(args.case_config)
    entities = {entity_id(case) for case in cases}
    if entities != EXPECTED_CASES:
        raise ValueError(f"case config must contain exactly the permanent seven cases; got {sorted(entities)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent / "inputs"
    target_root = ROOT / "configs" / "target_chirality"
    target_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.case_config)
    config_hash = sha256(config_path)
    rows = []
    for case in cases:
        case_entity = entity_id(case)
        target_path = target_root / f"{case_entity}.json"
        if not target_path.exists():
            target_path.write_text(json.dumps(target_config_for_case(case), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for seed_index, seed in enumerate(seeds, start=1):
            for condition, generator, prompt, d_ccd in (
                ("standard_af3", generate_baseline_input, False, False),
                ("at_af3", generate_prompt_conditioned_input, True, True),
            ):
                input_dir = root / condition / case_entity / f"seed_{seed}"
                input_dir.mkdir(parents=True, exist_ok=True)
                input_path = input_dir / "fold_input.json"
                input_path.write_text(json.dumps(generator(case, seed=seed), indent=2) + "\n", encoding="utf-8")
                rows.append({
                    "case_id": case["case_id"], "pdb_entity_id": case_entity, "pdb_id": case["pdb_id"],
                    "seed_index": seed_index, "inference_seed": seed, "condition": condition,
                    "input_json": manifest_path(input_path),
                    "output_dir": (Path(args.run_root) / condition / case_entity / f"seed_{seed}").as_posix(),
                    "target_config": manifest_path(target_path), "prompt_enabled": str(prompt).lower(),
                    "all_D_condition": str(not case_entity.startswith("8GQP")).lower(), "D_CCD_enabled": str(d_ccd).lower(),
                    "model_version": args.model_version, "code_commit": code_commit(), "config_sha256": config_hash, "status": "pending",
                })
    n_cases = len(cases)
    n_seeds = len(seeds)
    n_generation_conditions = len({row["condition"] for row in rows})
    n_generation_tasks = len(rows)
    assert n_cases == 7
    assert n_seeds == 30
    assert n_generation_conditions == 2
    assert n_generation_tasks == 420
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} fresh generation tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
