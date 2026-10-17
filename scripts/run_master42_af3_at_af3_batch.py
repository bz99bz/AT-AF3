"""Run only fresh master42 Standard-AF3 and AT-AF3 generation tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AF3_ROOT = Path(os.environ.get("AF3_ROOT", ROOT / "alphafold3"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_generation_manifest(rows: list[dict]) -> None:
    required = {"pdb_entity_id", "inference_seed", "condition"}
    missing = required.difference(rows[0]) if rows else required
    if missing:
        raise ValueError(f"generation manifest missing required fields: {sorted(missing)}")
    n_cases = len({row["pdb_entity_id"] for row in rows})
    n_seeds = len({int(row["inference_seed"]) for row in rows})
    n_generation_conditions = len({row["condition"] for row in rows})
    n_generation_tasks = len(rows)
    if not (n_cases == 7 and n_seeds == 30 and n_generation_conditions == 2 and n_generation_tasks == 420):
        raise ValueError("generation manifest must contain 7 cases x 30 seeds x 2 conditions = 420 tasks")


def output_complete(output: Path, row: dict) -> bool:
    metadata = output / "generation_metadata.json"
    return metadata.exists() and any(output.rglob("*.cif")) and json.loads(metadata.read_text(encoding="utf-8")).get("input_sha256") == sha256(project_path(row["input_json"]))


def resolve_af3_python(value: str | None) -> Path:
    """Prefer the project AF3 venv instead of the shell's Conda interpreter."""
    if value:
        return Path(value)
    env_python = os.environ.get("AF3_PYTHON")
    if env_python:
        return Path(env_python)
    project_venv = ROOT / "venv_af3" / "bin" / "python"
    return project_venv if project_venv.exists() else Path(sys.executable)


def validate_af3_runtime(python: Path) -> None:
    if not python.exists():
        raise FileNotFoundError(f"AF3 Python interpreter not found: {python}")
    check = subprocess.run(
        [str(python), "-c", "import absl, jax, sys; devices=jax.devices(); print('python=', sys.executable); print('jax_devices=', devices); print('has_gpu=', any(device.platform == 'gpu' for device in devices))"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "AF3 runtime preflight failed. Pass --af3-python /path/to/venv_af3/bin/python "
            "or set AF3_PYTHON.\n"
            + (check.stdout + check.stderr).strip()
        )
    if "has_gpu= True" not in check.stdout:
        raise RuntimeError("AF3 runtime preflight found no JAX GPU device:\n" + check.stdout.strip())
    print(check.stdout.strip())


def run_task(row: dict, args) -> tuple[str, str]:
    output = project_path(row["output_dir"])
    log = ROOT / args.log_root / row["condition"] / row["pdb_entity_id"] / f"seed_{row['inference_seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"start: {row['condition']} {row['pdb_entity_id']} seed={row['inference_seed']}", flush=True)
    if output.exists() and not args.resume:
        return row["pdb_entity_id"], "blocked_existing_output"
    if args.resume and output_complete(output, row):
        return row["pdb_entity_id"], "resumed_complete"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    input_json = project_path(row["input_json"])
    command = [str(args.af3_python), args.af3_script, f"--json_path={input_json.resolve()}", f"--output_dir={output.resolve()}", f"--model_dir={args.model_dir}"]
    if args.norun_data_pipeline:
        command.append("--norun_data_pipeline")
    started = datetime.now(timezone.utc).isoformat()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
    status = "success" if result.returncode == 0 and any(output.rglob("*.cif")) else "failed"
    metadata = {
        "case_id": row["pdb_entity_id"], "inference_seed": int(row["inference_seed"]), "seed_master": 42,
        "seed_set_name": "random_sample_master42_v1", "evaluation_round": "repeated_postcutoff_master42_v1",
        "condition": row["condition"], "af3_rerun": True,
        "prompt_enabled": row["prompt_enabled"] == "true", "all_D_condition": row["all_D_condition"] == "true",
        "D_CCD_enabled": row["D_CCD_enabled"] == "true", "chirality_correction": False,
        "reference_accessed_during_generation": False, "input_json": input_json.as_posix(), "input_sha256": sha256(input_json),
        "command": command, "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": result.returncode, "status": status, "log_path": str(log),
    }
    (output / "generation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row["pdb_entity_id"], status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fresh paired AF3 and AT-AF3 master42 tasks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--standard-output-root", default="runs_master42/standard_af3")
    parser.add_argument("--at-output-root", default="runs_master42/at_af3")
    parser.add_argument("--log-root", default="logs_master42")
    parser.add_argument("--af3-script", default=str(AF3_ROOT / "run_alphafold.py"))
    parser.add_argument("--af3-python", default=None, help="AF3 virtualenv Python; defaults to AF3_PYTHON, venv_af3/bin/python, or the current interpreter.")
    parser.add_argument("--model-dir", default=os.environ.get("AF3_MODEL_DIR", str(AF3_ROOT / "af3.bin")))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--norun-data-pipeline", action="store_true", default=True)
    args = parser.parse_args(argv)
    args.af3_python = resolve_af3_python(args.af3_python)
    rows = load_rows(Path(args.manifest))
    assert_generation_manifest(rows)
    selected = [r for r in rows if (not args.case_id or r["pdb_entity_id"] == args.case_id) and (args.seed is None or int(r["inference_seed"]) == args.seed)]
    if args.dry_run:
        for row in selected:
            print(f"DRY-RUN {row['condition']} {row['pdb_entity_id']} seed={row['inference_seed']} -> {row['output_dir']}")
        return 0
    validate_af3_runtime(args.af3_python)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(run_task, row, args) for row in selected]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"[{len(results)}/{len(selected)}] {results[-1][1]}: {results[-1][0]}", flush=True)
    failed = [result for result in results if result[1] not in {"success", "resumed_complete"}]
    print(f"Completed {len(results)} requested tasks; failures={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
