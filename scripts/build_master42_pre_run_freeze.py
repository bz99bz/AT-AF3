"""Write immutable provenance hashes before the master42 evaluation begins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AF3_ROOT = Path(os.environ.get("AF3_ROOT", ROOT / "alphafold3"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_digest(path: Path) -> str:
    return digest(path) if path.exists() else "not_present"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze master42 code/config provenance before inference.")
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--case-config", default="configs/at_af_7.json")
    parser.add_argument("--output", default="reports_master42_n30/pre_run_freeze_manifest.json")
    parser.add_argument("--run-root", default="runs_master42_n30")
    parser.add_argument("--af3-version", default="alphafold3_frozen")
    parser.add_argument("--database-version", default="not_recorded")
    parser.add_argument("--replace-before-run", action="store_true", help="Replace only a pre-run manifest before any master42 outputs exist.")
    args = parser.parse_args(argv)
    files = {
        "seed_manifest": ROOT / args.seed_file,
        "case_config": ROOT / args.case_config,
        "input_generation_code": ROOT / "scripts/generate_af3_inputs.py",
        "prompt_code": ROOT / "scripts/associative_chirality_memory.py",
        "ccd_mapping_code": ROOT / "scripts/utils.py",
        "chirality_geometry_code": ROOT / "src/chirality_geometry.py",
        "error_code": ROOT / "src/chirality_error_code.py",
        "local_mirror_code": ROOT / "src/local_chirality_mirror.py",
        "evaluation_code": ROOT / "scripts/evaluate_master42_experiment.py",
        "af3_entrypoint": AF3_ROOT / "run_alphafold.py",
    }
    payload = {
        "seed_set_name": "random_sample_master42_v1",
        "seed_master": 42,
        "evaluation_round": "repeated_postcutoff_master42_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "af3_version": args.af3_version,
        "database_version": args.database_version,
        "sha256": {name: optional_digest(path) for name, path in files.items()},
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            run_root = ROOT / args.run_root
            if not args.replace_before_run or run_root.exists():
                raise RuntimeError("pre-run freeze manifest exists but code/config hashes changed; start a new experiment round")
            payload["replaced_before_any_inference"] = True
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Replaced pre-run freeze manifest before inference at {output}")
            return 0
        print(f"Existing compatible freeze manifest retained at {output}")
        return 0
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote pre-run freeze manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
