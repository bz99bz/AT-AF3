"""Re-derive the comparison summary from existing run artifacts without re-detection.

The frozen detector and the per-model comparison already write every raw row; this module only
re-aggregates them with the summary logic in :func:`..run._summarise`, without re-reading or
re-hydrogenating a single structure.

What it adds to ``interaction_comparison_summary.csv``:

* ``outside_evaluable_domain``  model records whose partner residue is outside the mapped
  polymer domain;
* ``recall_identity`` / ``precision_identity`` / ``f1_identity``  pure identity retention, i.e.
  ``retained`` only, so geometric distortion is reported separately from interaction identity;
* ``reference_status``  ``compared`` / ``reference_only_model_absent`` / ``model_only_no_reference``;
* explicit rows for classes that exist in the reference but are entirely absent from the model
  (protein-metal coordination and water-mediated contacts: the models do not contain the ion or
  the waters), so an empty model class is retained explicitly.

Inputs (``--results-dir``):
  * ``interaction_comparison_per_model.csv``  raw retained/lost/extra/outside rows per model;
  * ``interactions.csv``                      every interaction record (GT rows = the reference).

Outputs:
  * ``interaction_comparison_summary_augmented.csv``
  * ``interaction_class_coverage.csv``  which (class, scope) exist in the reference and whether
    the model was able to produce them at all.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phenix_interaction_analysis import protocol as P  # noqa: E402
from phenix_interaction_analysis import run as RUN  # noqa: E402

DEFAULT_RESULTS = ROOT / "results/phenix_interaction"


def _read_rows(path: Path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def gt_reference_counts(interaction_rows: list) -> dict:
    """``{(target, dataset): {(class, scope): n}}`` over the ground-truth records."""
    counts: dict = {}
    for r in interaction_rows:
        if r.get("method") != P.METHOD_GT:
            continue
        by_cls = counts.setdefault((r["target"], r["dataset"]), {})
        key = (r["interaction_class"], r["scope"])
        by_cls[key] = by_cls.get(key, 0) + 1
    return counts


def coverage_rows(interaction_rows: list, gt_counts: dict) -> list:
    """One row per (class, scope): reference total, targets covered and model presence."""
    gt_total = defaultdict(int)
    gt_targets = defaultdict(set)
    for (target, _ds), by_cls in gt_counts.items():
        for key, n in by_cls.items():
            gt_total[key] += n
            gt_targets[key].add(target)
    model_total = defaultdict(int)
    for r in interaction_rows:
        if r.get("method") == P.METHOD_GT:
            continue
        model_total[(r["interaction_class"], r["scope"])] += 1
    keys = sorted(set(gt_total) | set(model_total))
    rows = []
    for cls, scope in keys:
        ref = gt_total.get((cls, scope), 0)
        mod = model_total.get((cls, scope), 0)
        rows.append({
            "interaction_class": cls, "scope": scope,
            "n_reference_total": ref, "n_targets_with_reference": len(gt_targets.get((cls, scope), ())),
            "n_model_total": mod,
            "evaluated": "yes" if (ref and mod) else ("reference_only" if ref else "model_only"),
            "note": "" if mod else "model contains no such interaction (ion/water absent)",
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=None,
                    help="output summary (default: <results-dir>/interaction_comparison_summary_augmented.csv)")
    args = ap.parse_args(argv)

    results = args.results_dir
    per_model = _read_rows(results / "interaction_comparison_per_model.csv")
    interactions = _read_rows(results / "interactions.csv")
    gt_counts = gt_reference_counts(interactions)

    summary = RUN._summarise(per_model, gt_counts)
    out = args.out or (results / "interaction_comparison_summary_augmented.csv")
    RUN.write_csv(out, summary, RUN.SUMMARY_COLS)

    cov = coverage_rows(interactions, gt_counts)
    cov_out = results / "interaction_class_coverage.csv"
    RUN.write_csv(cov_out, cov, ["interaction_class", "scope", "n_reference_total",
                                 "n_targets_with_reference", "n_model_total", "evaluated", "note"])

    print(f"wrote {out} ({len(summary)} rows)")
    print(f"wrote {cov_out} ({len(cov)} rows)")
    absent = sum(1 for r in summary if r["reference_status"] == P.REFERENCE_MODEL_ABSENT)
    print(f"reference_only_model_absent rows: {absent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
