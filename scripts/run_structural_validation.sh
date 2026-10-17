#!/usr/bin/env bash
# GT-referenced structural validation and the analyses used by the paper's
# structural-consistency panels.
#
# Run this only after both the All-D and Local-D arms have produced corrected
# structures. Phenix is required for canonicalisation, DSSP, CCTBX bond
# restraints and WGE-3. DockQ and USalign are optional external tools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
: "${PHENIX_PYTHON:?Set PHENIX_PYTHON to the Phenix interpreter}"
PHENIX_MOLPROBITY="${PHENIX_MOLPROBITY:-phenix.molprobity}"
export PHENIX_PYTHON PHENIX_MOLPROBITY

OUT="${GT_OUT:-results/gt_comparison_phenix_ccdfix}"
REPORTS="${REPORTS:-reports_master42_n30_ccdfix}"

"$PYTHON" scripts/run_gt_phenix_validation_linux.py \
  --output-dir "$OUT" \
  --phenix-python "$PHENIX_PYTHON" \
  --molprobity "$PHENIX_MOLPROBITY" \
  --all-d-standard-root runs_master42_n30/standard_af3 \
  --all-d-at-root runs_master42_n30/at_af3 \
  --all-d-corrected-root runs_master42_n30/at_af3_chirality_corrected

"$PHENIX_PYTHON" scripts/gt_metrics_suite.py \
  --canonical-dir "$OUT/canonical_pdb" \
  --target-config-root configs/target_chirality \
  --local-d-target-manifest data/local_d/local_d_curated_v1_1/14_correction_target_manifest.csv \
  --out-dir "$REPORTS"

"$PHENIX_PYTHON" scripts/tertiary_structure_rmsd.py \
  --canonical-dir "$OUT/canonical_pdb" \
  --out-dir "$REPORTS"
"$PHENIX_PYTHON" scripts/secondary_structure_q3.py \
  --canonical-dir "$OUT/canonical_pdb" \
  --out-dir "$REPORTS"

"$PHENIX_PYTHON" scripts/analyze_bond_geometry.py \
  --all-d-root runs_master42_n30/at_af3_chirality_corrected \
  --local-d-root runs/local_d_curated_v1_1_post_cutoff/corrected \
  --output-dir results/bond_geometry

PHENIX_REDUCE="${PHENIX_REDUCE:-phenix.reduce}"
"$PHENIX_PYTHON" scripts/phenix_interaction_analysis.py \
  --out-dir results/phenix_interaction \
  --canonical-dir "$OUT/canonical_pdb" \
  --chain-map "$OUT/canonical_chain_id_mapping.json" \
  --chirality-dir configs/target_chirality \
  --ccd-cache data/ccd_components_cache.json \
  --reduce "$PHENIX_REDUCE"

"$PYTHON" scripts/ccdfix_evaluable_table.py \
  --manifest "$OUT/gt_phenix_manifest.csv" \
  --target-config-root configs/target_chirality \
  --out "$REPORTS/table_all_d_chirality_evaluable.md"

"$PYTHON" scripts/locald_evaluable_table.py \
  --canonical-dir "$OUT/canonical_pdb" \
  --out "$REPORTS/table_locald_chirality.md"

echo "Structural validation complete: $OUT and $REPORTS"
