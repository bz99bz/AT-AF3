"""Seed-paired, post-generation evaluation for the frozen master42 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.chirality_error_code import build_error_code_rows, load_target_config
from src.chirality_geometry import clash_count
from src.structure_atom_loader import load_structure_atoms

EXPECTED_CASES = {"7P5R_2", "7QDI_1", "8AVR_2", "8BFD_2", "8GQP_1", "8JH6_1", "8P6Q_2"}
CLASH_DEFINITION = "inter-residue heavy-atom pairs with distance < 2.5 Angstrom; same residue and adjacent same-chain residues skipped"


def structure_path(directory: Path, corrected: bool = False) -> Path | None:
    if corrected:
        path = directory / "corrected_all_errors.cif"
        return path if path.exists() else None
    candidates = sorted(directory.rglob("*.cif")) + sorted(directory.rglob("*.pdb"))
    return candidates[0] if candidates else None


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metrics(path: Path, config: dict, case_id: str, seed: int) -> dict:
    atoms, meta = load_structure_atoms(path)
    if not meta.get("parse_success"):
        raise ValueError(meta.get("error", "parse failure"))
    rows = build_error_code_rows(atoms, config, case_id, seed)
    targets = [row for row in rows if row["target_chirality"] in {"D", "L"}]
    resolved = [row for row in targets if row["chirality_status"] == "resolved"]
    correct = [row for row in resolved if row["current_chirality"] == row["target_chirality"]]
    errors = [row for row in resolved if row["current_chirality"] != row["target_chirality"]]
    n_target = len(targets)
    accuracy = len(correct) / n_target if n_target else math.nan
    return {
        "n_target_chiral_residues": n_target, "n_correct_target_residues": len(correct),
        "n_errors": len(errors), "residual_error_fraction": len(errors) / n_target if n_target else math.nan,
        "n_unresolved": n_target - len(resolved), "target_D_accuracy": accuracy,
        "D90_indicator": int(math.isfinite(accuracy) and accuracy >= 0.90),
        "clash_count": clash_count(atoms), "clash_rate": clash_count(atoms) / max(len(atoms), 1),
    }


def bootstrap(values: list[float], rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    array = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not len(array):
        return math.nan, math.nan
    means = np.mean(rng.choice(array, size=(n_bootstrap, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate strictly paired master42 outputs after generation.")
    parser.add_argument("--standard-root", required=True)
    parser.add_argument("--at-root", required=True)
    parser.add_argument("--corrected-root", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--target-config-root", default="configs/target_chirality")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    seeds = [int(x) for x in json.loads(Path(args.seed_file).read_text(encoding="utf-8"))["inference_seeds"]]
    roots = {"standard_af3": Path(args.standard_root), "at_af3": Path(args.at_root), "at_af3_chirality_corrected": Path(args.corrected_root)}
    cases = sorted(path.stem for path in Path(args.target_config_root).glob("*.json"))
    if set(cases) != EXPECTED_CASES or len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("master42 n30 evaluation requires seven target configs and thirty frozen unique seeds")
    rows, quality_rows = [], []
    for case_id in cases:
        config = load_target_config(Path(args.target_config_root) / f"{case_id}.json")
        for seed_index, seed in enumerate(seeds, start=1):
            for condition, root in roots.items():
                directory = root / case_id / f"seed_{seed}"
                path = structure_path(directory, condition == "at_af3_chirality_corrected")
                if path is None:
                    raise FileNotFoundError(f"missing paired structure: {condition} {case_id} seed={seed}")
                result = metrics(path, config, case_id, seed)
                rows.append({"case_id": case_id, "seed_index": seed_index, "inference_seed": seed, "condition": condition, "structure_path": str(path), **result})
                quality_rows.append({"case_id": case_id, "inference_seed": seed, "condition": condition, "clash_count": result["clash_count"], "clash_rate": result["clash_rate"], "clash_definition": CLASH_DEFINITION, "backbone_RMSD": "not_applicable_without_pair", "CA_RMSD": "not_applicable_without_pair", "all_atom_RMSD": "not_applicable_without_pair", "TM_score_if_available": "not_available", "bond_length_outliers": "not_available", "bond_angle_outliers": "not_available", "peptide_planarity_outliers": "not_available", "non_target_chain_RMSD": "not_available", "interface_contact_change": "not_available"})
    rng = np.random.default_rng(args.bootstrap_seed)
    summary, paired = [], []
    by_key = {(r["case_id"], r["inference_seed"], r["condition"]): r for r in rows}
    metric_names = ["target_D_accuracy", "residual_error_fraction", "clash_count", "clash_rate"]
    comparisons = [("at_af3_minus_standard_af3", "at_af3", "standard_af3"), ("corrected_minus_at_af3", "at_af3_chirality_corrected", "at_af3"), ("corrected_minus_standard_af3", "at_af3_chirality_corrected", "standard_af3")]
    for case_id in cases:
        for condition in roots:
            subset = [r for r in rows if r["case_id"] == case_id and r["condition"] == condition]
            for metric in metric_names:
                values = [safe_float(r[metric]) for r in subset]
                low, high = bootstrap(values, rng, args.bootstrap_replicates)
                summary.append({"case_id": case_id, "condition": condition, "metric": metric, "n_seeds": len(values), "mean": float(np.nanmean(values)), "sample_SD": float(np.nanstd(values, ddof=1)), "median": float(np.nanmedian(values)), "min": float(np.nanmin(values)), "max": float(np.nanmax(values)), "bootstrap_95CI_low": low, "bootstrap_95CI_high": high, "D90_count_over_30": sum(r["D90_indicator"] for r in subset) if metric == "target_D_accuracy" else "not_applicable"})
        for label, left, right in comparisons:
            for metric in metric_names:
                deltas = [safe_float(by_key[(case_id, seed, left)][metric]) - safe_float(by_key[(case_id, seed, right)][metric]) for seed in seeds]
                low, high = bootstrap(deltas, rng, args.bootstrap_replicates)
                for seed, delta in zip(seeds, deltas):
                    paired.append({"case_id": case_id, "inference_seed": seed, "comparison": label, "metric": metric, "paired_delta": delta})
                summary.append({"case_id": case_id, "condition": label, "metric": metric, "n_seeds": len(deltas), "mean": float(np.nanmean(deltas)), "sample_SD": float(np.nanstd(deltas, ddof=1)), "median": float(np.nanmedian(deltas)), "min": float(np.nanmin(deltas)), "max": float(np.nanmax(deltas)), "bootstrap_95CI_low": low, "bootstrap_95CI_high": high, "D90_count_over_30": "not_applicable"})
    out = Path(args.output_root)
    write_csv(out / "seed_level_metrics.csv", rows)
    write_csv(out / "case_level_summary.csv", summary)
    write_csv(out / "paired_comparison_summary.csv", paired)
    write_csv(out / "geometry_quality_summary.csv", quality_rows)
    print(f"Wrote seed-level, paired bootstrap, and quality reports to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
