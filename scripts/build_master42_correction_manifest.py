"""Bind each deterministic correction task to its new master42 AT-AF3 parent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def find_structure(seed_dir: Path) -> Path | None:
    candidates = sorted(seed_dir.rglob("*.cif")) + sorted(seed_dir.rglob("*.pdb"))
    for path in candidates:
        if path.name not in {"corrected_all_errors.cif", "corrected_all_errors.pdb"}:
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the master42 deterministic-correction manifest.")
    parser.add_argument("--at-output-root", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--target-config-root", default="configs/target_chirality")
    parser.add_argument("--output", required=True)
    parser.add_argument("--correction-output-root", default="runs_master42_n30/at_af3_chirality_corrected")
    args = parser.parse_args(argv)
    seeds = json.loads(Path(args.seed_file).read_text(encoding="utf-8"))["inference_seeds"]
    if len(seeds) != 30:
        raise ValueError("expected exactly 30 master42 n30 seeds")
    root, targets, output = Path(args.at_output_root), Path(args.target_config_root), Path(args.output)
    rows = []
    for target in sorted(targets.glob("*.json")):
        case_id = target.stem
        for seed_index, seed in enumerate(seeds, start=1):
            parent_dir = root / case_id / f"seed_{seed}"
            structure = find_structure(parent_dir)
            metadata = parent_dir / "generation_metadata.json"
            if structure is None or not metadata.exists():
                raise FileNotFoundError(f"missing fresh AT-AF3 parent for {case_id} seed {seed}: {parent_dir}")
            parent = json.loads(metadata.read_text(encoding="utf-8"))
            if parent.get("condition") != "at_af3" or int(parent.get("inference_seed", -1)) != int(seed):
                raise ValueError(f"invalid parent provenance for {case_id} seed {seed}")
            rows.append({
                "case_id": case_id, "seed_index": seed_index, "inference_seed": seed,
                "at_af3_structure_path": manifest_path(structure), "target_config": manifest_path(target),
                "correction_output_dir": (Path(args.correction_output_root) / case_id / f"seed_{seed}").as_posix(),
                "input_structure_sha256": sha256(structure), "correction_mode": "all", "status": "pending",
            })
    n_cases = len(list(targets.glob("*.json")))
    n_seeds = len(seeds)
    n_correction_tasks = len(rows)
    assert n_cases == 7
    assert n_seeds == 30
    assert n_correction_tasks == 210
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} parent-bound correction tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
