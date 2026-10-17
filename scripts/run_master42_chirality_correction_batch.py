"""Apply deterministic correction only to structures enumerated in the master42 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_text(path: Path, n_lines: int = 40) -> str:
    if not path.exists():
        return "log_missing"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:]).strip()


def resolve_existing_path(value: str, label: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def assert_correction_manifest(rows: list[dict]) -> None:
    n_cases = len({row["case_id"] for row in rows})
    n_seeds = len({int(row["inference_seed"]) for row in rows})
    n_correction_tasks = len(rows)
    assert n_cases == 7
    assert n_seeds == 30
    assert n_correction_tasks == 210


def run_task(row: dict, args) -> tuple[str, str]:
    case_id, seed = row["case_id"], int(row["inference_seed"])
    output = ROOT / args.output_root / case_id / f"seed_{seed}"
    log = ROOT / args.log_root / "chirality_correction" / case_id / f"seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata.json"
    if metadata.exists() and args.resume:
        old = json.loads(metadata.read_text(encoding="utf-8"))
        if old.get("parent_structure_sha256") == row["input_structure_sha256"] and (output / "corrected_all_errors.cif").exists():
            return case_id, "resumed_complete"
    try:
        structure = resolve_existing_path(row["at_af3_structure_path"], "AT-AF3 parent structure")
        target_config = resolve_existing_path(row["target_config"], "target chirality config")
    except Exception as exc:
        log.write_text(str(exc) + "\n", encoding="utf-8")
        return case_id, f"failed_preflight:{exc}"
    command = [
        sys.executable,
        "scripts/run_full_chirality_correction.py",
        "--structure",
        str(structure),
        "--target-config",
        str(target_config),
        "--case-id",
        case_id,
        "--seed",
        str(seed),
        "--output-root",
        args.output_root,
        "--correction-mode",
        args.correction_mode,
    ]
    with log.open("w", encoding="utf-8") as handle:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0 or not (output / "corrected_all_errors.cif").exists():
        return case_id, "failed:" + tail_text(log, n_lines=12).replace("\n", " | ")
    metadata_payload = {
        "case_id": case_id, "inference_seed": seed, "seed_master": 42,
        "seed_set_name": "random_sample_master42_v1", "evaluation_round": "repeated_postcutoff_master42_v1",
        "condition": "at_af3_chirality_corrected", "parent_condition": "at_af3",
        "parent_structure_path": str(structure), "parent_structure_sha256": row["input_structure_sha256"],
        "af3_rerun": False, "correction_deterministic": True, "correction_mode": args.correction_mode,
        "reference_used_for_correction": False, "correction_log": str(log),
    }
    metadata.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return case_id, "success"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all parent-bound master42 chirality corrections.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="runs_master42/at_af3_chirality_corrected")
    parser.add_argument("--correction-mode", default="all", choices=["all"])
    parser.add_argument("--log-root", default="logs_master42")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert_correction_manifest(rows)
    rows = [
        row for row in rows
        if (not args.case_id or row["case_id"] == args.case_id)
        and (args.seed is None or int(row["inference_seed"]) == args.seed)
    ]
    if not rows:
        raise ValueError("no correction tasks selected")
    if args.dry_run:
        for row in rows:
            print(f"DRY-RUN correct {row['case_id']} seed={row['inference_seed']} parent={row['at_af3_structure_path']}")
        return 0
    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(run_task, row, args) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"{results[-1][1]}: {results[-1][0]}")
    failures = [result for result in results if result[1] not in {"success", "resumed_complete"}]
    print(f"Completed {len(results)} correction tasks; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
