from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "local_d" / "local_d_curated_v1_1"
RUN_ROOT = ROOT / "runs" / "local_d_curated_v1_1_post_cutoff"
VERSION = "local_d_curated_v1.1"
CONFIGURATION_VERSION = "local_d_curated_v1.1_post_cutoff_n30_v1"
MASTER_SEED = 42
N_SEEDS = 30
TARGET_CASES = [
    "7OOJ_1",
    "7P0A_3",
    "7P0T_3",
    "7W40_6",
    "8HR3_1",
    "8HR4_1",
    "8U78_1",
    "8UM9_1",
    "9BIC_1",
]
CONDITIONS = ["standard_af3", "at_af3"]
TARGET_MANIFEST = DATASET_DIR / "14_correction_target_manifest.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def seeds() -> list[int]:
    return random.Random(MASTER_SEED).sample(range(1, 2**31 - 1), N_SEEDS)


def load_case_map() -> dict[str, dict[str, str]]:
    rows = read_csv(DATASET_DIR / "06_local_d_case_manifest.csv")
    out = {}
    for r in rows:
        cid = r.get("case_id_normalized") or r.get("case_id")
        if cid in TARGET_CASES:
            out[cid] = r
    missing = sorted(set(TARGET_CASES) - set(out))
    if missing:
        raise RuntimeError(f"Missing target cases in case manifest: {missing}")
    return out


def build_input_jsons() -> dict[str, dict[str, str]]:
    comp_rows = [
        r for r in read_csv(DATASET_DIR / "22_complex_component_manifest.csv")
        if r["case_id"] in TARGET_CASES
    ]
    residue_rows = [
        r for r in read_csv(DATASET_DIR / "25_residue_identifier_mapping.csv")
        if r["case_id"] in TARGET_CASES and str(r.get("is_target_d_residue", "")).lower() == "true"
    ]
    by_case: dict[str, list[dict[str, str]]] = {c: [] for c in TARGET_CASES}
    for r in comp_rows:
        by_case[r["case_id"]].append(r)
    targets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in residue_rows:
        targets.setdefault((r["case_id"], r["sequence_chain_id"]), []).append(r)

    input_root = RUN_ROOT / "inputs"
    out: dict[str, dict[str, str]] = {}
    for case_id in TARGET_CASES:
        chains = sorted(by_case.get(case_id, []), key=lambda x: int(x.get("component_index") or 0))
        if not chains:
            raise RuntimeError(f"No complex components for {case_id}")
        standard_sequences = []
        at_sequences = []
        for ch in chains:
            chain_id = ch["asym_id"]
            standard_sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": ch["sequence"],
                    "modifications": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            })
            mods = []
            for rr in sorted(targets.get((case_id, chain_id), []), key=lambda x: int(x["af3_input_residue_index"])):
                mods.append({
                    "ptmType": rr["ccd_id"],
                    "ptmPosition": int(rr["af3_input_residue_index"]),
                })
            at_sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": ch["sequence"],
                    "modifications": mods,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            })
        for condition, name_suffix, sequences in [
            ("standard_af3", "standard_af3", standard_sequences),
            ("at_af3", "at_af3", at_sequences),
        ]:
            payload = {
                "name": f"{case_id}_{name_suffix}",
                "modelSeeds": [MASTER_SEED],
                "sequences": sequences,
                "dialect": "alphafold3",
                "version": 2,
            }
            subdir = "standard" if condition == "standard_af3" else "at_af3"
            path = input_root / subdir / f"{case_id}_{'standard' if condition == 'standard_af3' else 'at_af3'}.json"
            write_json(path, payload)
            out.setdefault(case_id, {})[condition] = rel(path)
    return out


def primary_cif_path(output_dir: Path, input_name: str) -> Path:
    return output_dir / input_name / f"{input_name}_model.cif"


def generation_job_id(case_id: str, condition: str, seed: int) -> str:
    return f"locald_postcutoff_v1__{case_id}__{condition}__seed_{seed}"


def correction_job_id(case_id: str, seed: int) -> str:
    return f"locald_postcutoff_v1__{case_id}__correction__seed_{seed}"


def build_generation_manifest(input_paths: dict[str, dict[str, str]], case_map: dict[str, dict[str, str]]) -> list[dict]:
    rows = []
    for case_id in TARGET_CASES:
        case = case_map[case_id]
        for seed_index, seed in enumerate(seeds(), 1):
            for condition in CONDITIONS:
                input_path = ROOT / input_paths[case_id][condition]
                input_name = f"{case_id}_{condition}"
                output_kind = "standard_af3" if condition == "standard_af3" else "at_af3_first_pass"
                output_dir = RUN_ROOT / output_kind / case_id / f"seed_{seed}"
                rows.append({
                    "dataset_version": VERSION,
                    "case_id": case_id,
                    "pdb_id": case.get("pdb_id_normalized") or case.get("pdb_id"),
                    "entity_id": case.get("entity_id", ""),
                    "split": "locked_post_cutoff_candidate",
                    "dataset_role": "post_cutoff_local_d_test",
                    "evaluation_round": "local_d_post_cutoff_n30",
                    "configuration_version": CONFIGURATION_VERSION,
                    "condition": condition,
                    "seed_index": seed_index,
                    "seed": seed,
                    "input_json_path": input_paths[case_id][condition],
                    "input_json_sha256": sha256(input_path),
                    "run_status": "not_started",
                    "allowed_to_run": "True",
                    "blocked_reason": "",
                    "expected_first_pass_output": rel(primary_cif_path(output_dir, input_name)),
                    "expected_log_path": rel(RUN_ROOT / "logs" / condition / case_id / f"seed_{seed}.log"),
                    "generation_job_id": generation_job_id(case_id, condition, seed),
                    "master_seed": MASTER_SEED,
                    "seed_generation_method": "random.Random(42).sample(range(1, 2**31 - 1), 30)",
                    "output_dir": rel(output_dir),
                    "expected_metadata_path": rel(RUN_ROOT / "metadata" / condition / case_id / f"seed_{seed}" / "generation_metadata.json"),
                })
    return rows


def build_correction_manifest(generation_rows: list[dict]) -> list[dict]:
    at_rows = [r for r in generation_rows if r["condition"] == "at_af3"]
    rows = []
    for r in at_rows:
        case_id = r["case_id"]
        seed = int(r["seed"])
        out_dir = RUN_ROOT / "corrected" / case_id / f"seed_{seed}"
        rows.append({
            "correction_job_id": correction_job_id(case_id, seed),
            "case_id": case_id,
            "seed": seed,
            "source_at_af3_generation_job_id": r["generation_job_id"],
            "at_af3_first_pass_output_path": r["expected_first_pass_output"],
            "correction_target_manifest_path": rel(TARGET_MANIFEST),
            "correction_status": "not_started",
            "correction_deterministic": "True",
            "reference_used_for_correction": "False",
            "error_code_status": "compute_from_prediction_at_runtime",
            "corrected_output_path": rel(out_dir / "corrected_all_errors.cif"),
            "metadata_output_path": rel(out_dir / "correction_summary.json"),
            "dataset_version": VERSION,
            "configuration_version": CONFIGURATION_VERSION,
            "master_seed": MASTER_SEED,
            "seed_generation_method": "random.Random(42).sample(range(1, 2**31 - 1), 30)",
            "allowed_to_run": "True",
            "blocked_reason": "",
        })
    return rows


def build_status(generation_rows: list[dict], correction_rows: list[dict]) -> None:
    gen_status = []
    for r in generation_rows:
        gen_status.append({
            "generation_job_id": r["generation_job_id"],
            "case_id": r["case_id"],
            "condition": r["condition"],
            "seed": r["seed"],
            "status": "not_started",
            "attempt_count": "0",
            "start_time": "",
            "end_time": "",
            "exit_code": "",
            "input_json_sha256": r["input_json_sha256"],
            "output_exists": "False",
            "metadata_exists": "False",
            "log_exists": "False",
            "error_type": "",
            "error_message": "",
        })
    corr_status = []
    for r in correction_rows:
        corr_status.append({
            "correction_job_id": r["correction_job_id"],
            "case_id": r["case_id"],
            "seed": r["seed"],
            "status": "not_started",
            "attempt_count": "0",
            "start_time": "",
            "end_time": "",
            "exit_code": "",
            "source_at_af3_generation_job_id": r["source_at_af3_generation_job_id"],
            "corrected_output_exists": "False",
            "metadata_exists": "False",
            "log_exists": "False",
            "error_type": "",
            "error_message": "",
        })
    write_csv(RUN_ROOT / "status" / "post_cutoff_generation_status.csv", gen_status)
    write_csv(RUN_ROOT / "status" / "post_cutoff_correction_status.csv", corr_status)


def write_run_files(generation_rows: list[dict], correction_rows: list[dict]) -> None:
    for sub in ["manifests", "scripts", "status", "reports", "metadata", "logs", "standard_af3", "at_af3_first_pass", "corrected"]:
        (RUN_ROOT / sub).mkdir(parents=True, exist_ok=True)
    write_csv(RUN_ROOT / "manifests" / "post_cutoff_generation_manifest.csv", generation_rows)
    write_csv(RUN_ROOT / "manifests" / "post_cutoff_correction_manifest.csv", correction_rows)
    build_status(generation_rows, correction_rows)
    summary = {
        "created_at_utc": now(),
        "dataset_version": VERSION,
        "configuration_version": CONFIGURATION_VERSION,
        "case_ids": TARGET_CASES,
        "n_cases": len(TARGET_CASES),
        "n_seeds": N_SEEDS,
        "n_conditions": 2,
        "n_generation_tasks": len(generation_rows),
        "n_correction_tasks": len(correction_rows),
        "n_total_tasks": len(generation_rows) + len(correction_rows),
        "master_seed": MASTER_SEED,
        "inference_seeds": seeds(),
        "reference_used_for_correction": False,
        "workflow": "9 post-cutoff Local-D cases x 30 seeds x (Standard AF3 + AT-AF3 + deterministic correction)",
    }
    write_json(RUN_ROOT / "metadata" / "post_cutoff_workflow_manifest.json", summary)
    report = [
        "# Local-D post-cutoff n30 workflow",
        "",
        f"- cases: `{len(TARGET_CASES)}`",
        f"- seeds: `{N_SEEDS}`",
        f"- generation tasks: `{len(generation_rows)}`",
        f"- correction tasks: `{len(correction_rows)}`",
        f"- total tasks: `{len(generation_rows) + len(correction_rows)}`",
        "",
        "## Cases",
        "",
        *[f"- `{c}`" for c in TARGET_CASES],
    ]
    (RUN_ROOT / "reports" / "post_cutoff_workflow_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    case_map = load_case_map()
    input_paths = build_input_jsons()
    generation_rows = build_generation_manifest(input_paths, case_map)
    correction_rows = build_correction_manifest(generation_rows)
    if len(generation_rows) != 540:
        raise RuntimeError(f"Expected 540 generation rows, got {len(generation_rows)}")
    if len(correction_rows) != 270:
        raise RuntimeError(f"Expected 270 correction rows, got {len(correction_rows)}")
    write_run_files(generation_rows, correction_rows)
    print(json.dumps({
        "run_root": rel(RUN_ROOT),
        "cases": TARGET_CASES,
        "generation_tasks": len(generation_rows),
        "correction_tasks": len(correction_rows),
        "total_tasks": len(generation_rows) + len(correction_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
