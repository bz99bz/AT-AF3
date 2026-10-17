#!/usr/bin/env python3
"""Build the measured CSV inputs for redesign_covalent_geometry.py.

Reads the project's bond-class analysis outputs and writes, into this kit:

- bond_classes.csv       per-task normalized deviation from GT (%) for AF3 and
                         Corrected AT-AF3, one row per task x bond class.
                         Deviation = |condition mean - GT mean| / GT mean,
                         where means are the dataset-level observed mean bond
                         lengths (mean over structures of the per-structure
                         class mean).
- correction_changes.csv per structure x bond class absolute normalized
                         bond-length change during the correction step,
                         |Corrected - AT-AF3| / AT-AF3 (%).  Zeros are kept.
- overall_summary_measured.csv
                         overall mean normalized deviation per task, computed
                         the way the source figure does: average the 11 class
                         mean lengths per condition, then take the normalized
                         deviation between the condition and GT averages.  This
                         is NOT the average of the 11 per-class deviations.

Validated: the All-D class means reproduce the hand-supplied table exactly
(e.g. C=O AF3 1.23215, GT 1.23099) and the All-D overall deviations reproduce
0.0405% / 0.9122%.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
DEFAULT_DATASET_SUMMARY = KIT / "../../results/bond_class_mean_overlap/bond_class_mean_dataset_summary.csv"
DEFAULT_PER_STRUCTURE = KIT / "../../results/bond_class_mean_overlap/bond_class_mean_per_structure.csv"

# (bond_class id used by the plotting script, bond_type, environment)
CLASSES = [
    ("n_ca_pro", "CA-N", "PRO"),
    ("n_ca_gly", "CA-N", "GLY"),
    ("n_ca_other", "CA-N", "other"),
    ("c_o_all", "C=O", "all"),
    ("ca_c_gly", "C-CA", "GLY"),
    ("ca_c_other", "C-CA", "other"),
    ("peptide_c_n_xpro", "peptide C-N", "X-PRO"),
    ("peptide_c_n_nonxpro", "peptide C-N", "non-X-PRO"),
    ("ca_cb_ala", "CA-CB", "ALA"),
    ("ca_cb_iltval", "CA-CB", "ILE_THR_VAL"),
    ("ca_cb_other", "CA-CB", "other"),
]
TASKS = ("All-D", "Local-D")
CONDITIONS = ("GT", "AF3", "AT-AF3", "Corrected AT-AF3")


def read_dataset_means(path: Path) -> dict:
    """(task, class_id, condition) -> dataset-level observed mean length (A)."""
    means = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in CONDITIONS:
                continue
            for class_id, bond_type, environment in CLASSES:
                if row["bond_type"] == bond_type and row["environment"] == environment:
                    key = (row["dataset_type"], class_id, row["condition"])
                    if key in means:
                        raise ValueError(f"duplicate dataset summary row for {key}")
                    means[key] = float(row["observed_mean_A_mean"])
    missing = [(t, c, cond) for t in TASKS for c, _, _ in CLASSES for cond in CONDITIONS
               if (t, c, cond) not in means]
    if missing:
        raise ValueError(f"dataset summary is missing {len(missing)} task/class/condition "
                         f"combinations, e.g. {missing[:3]}")
    return means


def write_bond_classes(path: Path, means: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "bond_class", "af3_pct", "corrected_pct"])
        for task in TASKS:
            for class_id, _, _ in CLASSES:
                gt = means[(task, class_id, "GT")]
                af3 = abs(means[(task, class_id, "AF3")] - gt) / gt * 100
                corrected = abs(means[(task, class_id, "Corrected AT-AF3")] - gt) / gt * 100
                writer.writerow([task, class_id, f"{af3:.6f}", f"{corrected:.6f}"])


def write_summary(path: Path, means: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "af3_pct", "corrected_pct"])
        for task in TASKS:
            gt = statistics.fmean(means[(task, c, "GT")] for c, _, _ in CLASSES)
            af3 = statistics.fmean(means[(task, c, "AF3")] for c, _, _ in CLASSES)
            corrected = statistics.fmean(means[(task, c, "Corrected AT-AF3")] for c, _, _ in CLASSES)
            writer.writerow([task, f"{abs(af3 - gt) / gt * 100:.6f}",
                             f"{abs(corrected - gt) / gt * 100:.6f}"])


def write_changes(path: Path, per_structure: Path) -> int:
    paired: dict[tuple, dict[str, float]] = defaultdict(dict)
    with per_structure.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in ("AT-AF3", "Corrected AT-AF3") or not row["observed_mean_A"]:
                continue
            key = (row["dataset_type"], row["case"], row["seed"],
                   row["bond_type"], row["environment"])
            paired[key][row["condition"]] = float(row["observed_mean_A"])
    n = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction_id", "task", "target", "seed", "change_pct"])
        for (task, case, seed, bond_type, environment), values in sorted(paired.items()):
            if "AT-AF3" not in values or "Corrected AT-AF3" not in values:
                continue
            before = values["AT-AF3"]
            change = abs(values["Corrected AT-AF3"] - before) / before * 100
            class_id = next(c for c, bt, env in CLASSES if bt == bond_type and env == environment)
            writer.writerow([f"{case}_seed{seed}_{class_id}", task, case, seed,
                             f"{change:.10f}"])
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-summary", type=Path, default=DEFAULT_DATASET_SUMMARY)
    parser.add_argument("--per-structure", type=Path, default=DEFAULT_PER_STRUCTURE)
    parser.add_argument("--out-dir", type=Path, default=KIT)
    args = parser.parse_args()

    means = read_dataset_means(args.dataset_summary.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bonds = args.out_dir / "bond_classes.csv"
    summary = args.out_dir / "overall_summary_measured.csv"
    changes = args.out_dir / "correction_changes.csv"
    write_bond_classes(bonds, means)
    write_summary(summary, means)
    n = write_changes(changes, args.per_structure.resolve())
    print(f"wrote {bonds} (22 rows), {summary}, {changes} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
