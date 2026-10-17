from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from permanent_test_guard import is_permanent_test_case
from src.chirality_error_code import build_error_code_rows, load_target_config, write_error_code_outputs
from src.chirality_geometry import ensure_chirality_sign_mapping, geometry_quality
from src.local_chirality_mirror import apply_error_code_file, classify_after_for_rows, write_cif, write_pdb
from src.structure_atom_loader import load_structure_atoms


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run binary chirality correction for one explicitly supplied structure.")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--correction-mode", default="all")
    parser.add_argument("--output-root", default="chirality_correction_outputs")
    parser.add_argument("--max-corrections", type=int, default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_root) / args.case_id / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_permanent_test_case(args.case_id):
        (out_dir / "permanent_postcutoff_notice.json").write_text(
            json.dumps({
                "case_id": args.case_id,
                "notice": "explicit single-case transform only; no automatic permanent post-cutoff batch run",
            }, indent=2),
            encoding="utf-8",
        )

    ensure_chirality_sign_mapping(out_dir / "chirality_sign_mapping.json")
    atoms, meta = load_structure_atoms(args.structure)
    if not meta.get("parse_success"):
        raise SystemExit(f"Failed to parse structure: {meta.get('error')}")
    config = load_target_config(args.target_config, case_id=args.case_id)
    rows = build_error_code_rows(atoms, config, args.case_id, args.seed)
    package = write_error_code_outputs(rows, out_dir, args.case_id, args.seed)
    shutil.copy2(args.structure, out_dir / f"first_pass_input{Path(args.structure).suffix or '.pdb'}")

    corrected, correction_stats, metadata = apply_error_code_file(
        atoms,
        out_dir / "chirality_error_code.json",
        correction_mode=args.correction_mode,
        max_corrections=args.max_corrections,
    )
    write_cif(corrected, out_dir / "corrected_all_errors.cif", data_name=args.case_id)
    write_pdb(corrected, out_dir / "corrected_all_errors.pdb", remarks=[
        "correction_method=binary_chirality_error_code_plus_local_mirror",
        "correction_mode=all",
        "reference_used_for_correction=false",
        "target_chirality_source=input_condition_or_CCD",
    ])
    per_residue = classify_after_for_rows(atoms, corrected, rows, correction_stats)
    quality = geometry_quality(atoms, corrected, {
        (r["chain_id"], int(r["residue_index"])) for r in rows if r["error_code"] == 1 and r["correction_eligible"]
    })
    write_csv(out_dir / "correction_per_residue.csv", per_residue)
    write_csv(out_dir / "geometry_quality_before_after.csv", [quality])
    summary = {
        "case_id": args.case_id,
        "seed": args.seed,
        "correction_method": "binary_chirality_error_code_plus_local_mirror",
        "correction_mode": args.correction_mode,
        "reference_used_for_correction": False,
        "target_chirality_source": "input_condition_or_CCD",
        "n_error_code_1": sum(1 for r in rows if r["error_code"] == 1),
        "n_correction_eligible": sum(1 for r in rows if r["correction_eligible"]),
        "n_successfully_corrected": sum(1 for r in per_residue if r["correction_status"] == "successfully_corrected"),
        "n_correct_residues_damaged": sum(1 for r in per_residue if r["correction_status"] == "correct_residue_damaged"),
        "correction_package": package,
        "residue_correction_stats": correction_stats,
        **metadata,
    }
    (out_dir / "correction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote full chirality correction outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
