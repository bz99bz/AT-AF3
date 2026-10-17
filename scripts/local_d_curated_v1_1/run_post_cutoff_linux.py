from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AF3_ROOT = Path(os.environ.get("AF3_ROOT", ROOT / "alphafold3"))
RUN_ROOT = ROOT / "runs" / "local_d_curated_v1_1_post_cutoff"
GEN_MANIFEST = RUN_ROOT / "manifests" / "post_cutoff_generation_manifest.csv"
CORR_MANIFEST = RUN_ROOT / "manifests" / "post_cutoff_correction_manifest.csv"
GEN_STATUS = RUN_ROOT / "status" / "post_cutoff_generation_status.csv"
CORR_STATUS = RUN_ROOT / "status" / "post_cutoff_correction_status.csv"
TARGET_CASES = {
    "7OOJ_1", "7P0A_3", "7P0T_3", "7W40_6", "8HR3_1", "8HR4_1", "8U78_1", "8UM9_1", "9BIC_1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def update_status(path: Path, id_column: str, job_id: str, updates: dict[str, str]) -> None:
    rows = read_csv(path)
    matched = False
    for r in rows:
        if r[id_column] == job_id:
            r.update({k: str(v) for k, v in updates.items()})
            matched = True
            break
    if not matched:
        raise RuntimeError(f"status row not found: {job_id}")
    write_csv(path, rows)


def env_check(require_gpu: bool = True) -> None:
    af3_python = os.environ.get("AF3_PYTHON", "")
    model_dir = os.environ.get("AF3_MODEL_DIR", "")
    if not af3_python or not Path(af3_python).is_file():
        raise RuntimeError(f"AF3_PYTHON invalid: {af3_python}")
    if not os.access(af3_python, os.X_OK):
        raise RuntimeError(f"AF3_PYTHON not executable: {af3_python}")
    if not model_dir or not Path(model_dir).exists():
        raise RuntimeError(f"AF3_MODEL_DIR invalid: {model_dir}")
    if require_gpu:
        p = subprocess.run(
            [af3_python, "-c", "import jax; print(jax.default_backend())"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if p.returncode != 0 or p.stdout.strip().splitlines()[-1:] != ["gpu"]:
            raise RuntimeError(f"JAX GPU backend unavailable: stdout={p.stdout} stderr={p.stderr}")


def filtered_generation_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [r for r in read_csv(GEN_MANIFEST) if r["allowed_to_run"] == "True"]
    if args.case_id:
        rows = [r for r in rows if r["case_id"] == args.case_id]
    if args.seed:
        rows = [r for r in rows if int(r["seed"]) == args.seed]
    if args.condition:
        rows = [r for r in rows if r["condition"] == args.condition]
    for r in rows:
        if r["case_id"] not in TARGET_CASES:
            raise RuntimeError(f"unexpected case in post-cutoff generation manifest: {r['case_id']}")
    return rows


def filtered_correction_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [r for r in read_csv(CORR_MANIFEST) if r["allowed_to_run"] == "True"]
    if args.case_id:
        rows = [r for r in rows if r["case_id"] == args.case_id]
    if args.seed:
        rows = [r for r in rows if int(r["seed"]) == args.seed]
    for r in rows:
        if r["case_id"] not in TARGET_CASES:
            raise RuntimeError(f"unexpected case in post-cutoff correction manifest: {r['case_id']}")
    return rows


def find_primary_cif(output_dir: Path, expected: Path | None = None) -> Path | None:
    if expected and expected.exists():
        return expected
    if not output_dir.exists():
        return None
    cifs = sorted(output_dir.rglob("*.cif"), key=lambda p: (len(p.relative_to(output_dir).parts), "sample-" in str(p), str(p)))
    return cifs[0] if cifs else None


def execute_generation(row: dict[str, str], resume: bool) -> dict:
    af3_python = os.environ["AF3_PYTHON"]
    model_dir = os.environ["AF3_MODEL_DIR"]
    out_dir = ROOT / row["output_dir"]
    expected = ROOT / row["expected_first_pass_output"]
    meta_path = ROOT / row["expected_metadata_path"]
    log_path = ROOT / row["expected_log_path"]
    primary = find_primary_cif(out_dir, expected)
    if resume and meta_path.exists() and primary:
        return {"status": "success", "return_code": 0, "skipped": True, "primary_cif": rel(primary)}
    if out_dir.exists() and not resume:
        raise RuntimeError(f"output exists and --no-resume was requested: {out_dir}")
    template_path = ROOT / row["input_json_path"]
    obj = json.loads(template_path.read_text(encoding="utf-8"))
    obj["modelSeeds"] = [int(row["seed"])]
    generated_input = RUN_ROOT / "metadata" / "generated_inputs" / row["condition"] / row["case_id"] / f"seed_{row['seed']}" / "fold_input.json"
    write_json(generated_input, obj)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    start = now()
    update_status(GEN_STATUS, "generation_job_id", row["generation_job_id"], {
        "status": "running",
        "start_time": start,
        "error_type": "",
        "error_message": "",
    })
    cmd = [
        af3_python,
        str(AF3_ROOT / "run_alphafold.py"),
        f"--json_path={generated_input}",
        f"--output_dir={out_dir}",
        f"--model_dir={model_dir}",
        "--norun_data_pipeline",
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log_path.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    end = now()
    primary = find_primary_cif(out_dir, expected)
    status = "success" if p.returncode == 0 and primary else "failed"
    meta = {
        "generation_job_id": row["generation_job_id"],
        "case_id": row["case_id"],
        "condition": row["condition"],
        "seed": int(row["seed"]),
        "input_json_template": row["input_json_path"],
        "generated_input_json": rel(generated_input),
        "output_dir": row["output_dir"],
        "primary_cif_path": rel(primary) if primary else "",
        "log_path": row["expected_log_path"],
        "started_at_utc": start,
        "ended_at_utc": end,
        "exit_code": p.returncode,
        "status": status,
        "correction_executed": False,
        "reference_accessed": False,
    }
    write_json(meta_path, meta)
    update_status(GEN_STATUS, "generation_job_id", row["generation_job_id"], {
        "status": status,
        "end_time": end,
        "exit_code": p.returncode,
        "output_exists": str(bool(primary)),
        "metadata_exists": str(meta_path.exists()),
        "log_exists": str(log_path.exists()),
        "error_type": "" if status == "success" else "AF3_FAILED_OR_NO_CIF",
        "error_message": "" if status == "success" else f"See log: {log_path}",
    })
    return {"status": status, "return_code": p.returncode, "skipped": False, "primary_cif": rel(primary) if primary else ""}


def execute_correction(row: dict[str, str], resume: bool) -> dict:
    correction_python = os.environ.get("AT_AF_PYTHON") or os.environ.get("AF3_PYTHON") or sys.executable
    corrected = ROOT / row["corrected_output_path"]
    meta = ROOT / row["metadata_output_path"]
    if resume and corrected.exists() and meta.exists():
        return {"status": "success", "return_code": 0, "skipped": True, "corrected": rel(corrected)}
    source = ROOT / row["at_af3_first_pass_output_path"]
    if not source.exists():
        source = find_primary_cif(RUN_ROOT / "at_af3_first_pass" / row["case_id"] / f"seed_{row['seed']}", None)
    if not source or not source.exists():
        raise RuntimeError(f"AT-AF3 first-pass source missing for correction: {row['case_id']} seed {row['seed']}")
    log_path = RUN_ROOT / "logs" / "correction" / row["case_id"] / f"seed_{row['seed']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = now()
    update_status(CORR_STATUS, "correction_job_id", row["correction_job_id"], {
        "status": "running",
        "start_time": start,
        "error_type": "",
        "error_message": "",
    })
    cmd = [
        correction_python,
        str(ROOT / "scripts" / "run_full_chirality_correction.py"),
        "--structure", str(source),
        "--target-config", str(ROOT / row["correction_target_manifest_path"]),
        "--case-id", row["case_id"],
        "--seed", str(row["seed"]),
        "--output-root", str(RUN_ROOT / "corrected"),
        "--correction-mode", "all",
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log_path.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    end = now()
    status = "success" if p.returncode == 0 and corrected.exists() else "failed"
    update_status(CORR_STATUS, "correction_job_id", row["correction_job_id"], {
        "status": status,
        "end_time": end,
        "exit_code": p.returncode,
        "corrected_output_exists": str(corrected.exists()),
        "metadata_exists": str(meta.exists()),
        "log_exists": str(log_path.exists()),
        "error_type": "" if status == "success" else "CORRECTION_FAILED",
        "error_message": "" if status == "success" else f"See log: {log_path}",
    })
    return {"status": status, "return_code": p.returncode, "skipped": False, "corrected": rel(corrected) if corrected.exists() else ""}


def completed_durations(status_path: Path) -> list[float]:
    durations = []
    for r in read_csv(status_path):
        if r.get("status") in {"success", "failed"}:
            s = parse_time(r.get("start_time", ""))
            e = parse_time(r.get("end_time", ""))
            if s and e and e >= s:
                durations.append((e - s).total_seconds())
    return durations


def progress(label: str, completed: int, total: int, started_mono: float, extra_durations: list[float] | None = None) -> None:
    elapsed = time.monotonic() - started_mono
    durations = list(extra_durations or [])
    if completed:
        durations.append(elapsed / max(completed, 1))
    avg = sum(durations) / len(durations) if durations else 0
    remaining = max(total - completed, 0)
    eta = timedelta(seconds=int(avg * remaining)) if avg > 0 else "unknown"
    pct = 100.0 * completed / total if total else 100.0
    print(f"[{now()}] {label}: {completed}/{total} ({pct:.1f}%) remaining={remaining} avg_sec={avg:.1f} eta={eta}", flush=True)


def write_final_report(payload: dict) -> None:
    write_json(RUN_ROOT / "metadata" / "post_cutoff_run_result.json", payload)
    lines = [
        "# Local-D post-cutoff n30 run report",
        "",
        f"- phase: `{payload['phase']}`",
        f"- generation_success: `{payload['generation_success']}`",
        f"- generation_failures: `{payload['generation_failures']}`",
        f"- correction_success: `{payload['correction_success']}`",
        f"- correction_failures: `{payload['correction_failures']}`",
        "",
        "## Failure examples",
    ]
    for f in payload["failures"][:50]:
        lines.append(f"- `{f['job_id']}`: {f['error']}")
    (RUN_ROOT / "reports" / "post_cutoff_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generation", "correction", "all"], default="all")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=["standard_af3", "at_af3"], default="")
    args = parser.parse_args(argv)

    if args.phase in {"generation", "all"}:
        env_check(require_gpu=True)
    elif args.phase == "correction":
        if not (os.environ.get("AT_AF_PYTHON") or os.environ.get("AF3_PYTHON")):
            raise RuntimeError("Set AT_AF_PYTHON or AF3_PYTHON before running correction")

    gen_rows = filtered_generation_rows(args)
    corr_rows = filtered_correction_rows(args)
    if not args.case_id and not args.seed and not args.condition:
        if len(gen_rows) != 540:
            raise RuntimeError(f"Expected 540 generation tasks, got {len(gen_rows)}")
        if len(corr_rows) != 270:
            raise RuntimeError(f"Expected 270 correction tasks, got {len(corr_rows)}")

    failures = []
    generation_success = 0
    correction_success = 0
    started = now()

    if args.phase in {"generation", "all"}:
        phase_start = time.monotonic()
        total = len(gen_rows)
        for i, row in enumerate(gen_rows, 1):
            try:
                result = execute_generation(row, resume=args.resume)
                if result["status"] == "success":
                    generation_success += 1
                else:
                    failures.append({"job_id": row["generation_job_id"], "error": result})
            except Exception as exc:
                failures.append({"job_id": row["generation_job_id"], "error": repr(exc)})
            progress("generation", i, total, phase_start, completed_durations(GEN_STATUS))

    if args.phase in {"correction", "all"}:
        phase_start = time.monotonic()
        total = len(corr_rows)
        for i, row in enumerate(corr_rows, 1):
            try:
                result = execute_correction(row, resume=args.resume)
                if result["status"] == "success":
                    correction_success += 1
                else:
                    failures.append({"job_id": row["correction_job_id"], "error": result})
            except Exception as exc:
                failures.append({"job_id": row["correction_job_id"], "error": repr(exc)})
            progress("correction", i, total, phase_start, completed_durations(CORR_STATUS))

    payload = {
        "phase": args.phase,
        "resume": args.resume,
        "started_at_utc": started,
        "ended_at_utc": now(),
        "generation_total": len(gen_rows),
        "correction_total": len(corr_rows),
        "generation_success": generation_success,
        "correction_success": correction_success,
        "generation_failures": sum(1 for f in failures if "__correction__" not in f["job_id"]),
        "correction_failures": sum(1 for f in failures if "__correction__" in f["job_id"]),
        "failures": failures,
        "reference_accessed": False,
    }
    write_final_report(payload)
    print(json.dumps({
        "generation_success": generation_success,
        "correction_success": correction_success,
        "n_failures": len(failures),
    }, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
