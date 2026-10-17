#!/usr/bin/env bash
# Reproduce the All-D arm of CHORD (7 targets x 30 matched seeds).
#
# Required environment:
#   AF3_PYTHON      Python from the AlphaFold 3 environment
#   AF3_MODEL_DIR   directory/file accepted by AF3 --model_dir
#   AF3_ROOT        checkout containing run_alphafold.py (default: <repo>/alphafold3)
#
# Optional:
#   PYTHON          interpreter for the AT-AF3 scripts (default: python)
#   MAX_PARALLEL    concurrent AF3 jobs (default: 1)
#   FORCE=1         rebuild the input/manifest tree before running
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${MANIFEST:-manifests_master42_n30/master42_generation_manifest.csv}"
CORRECTION_MANIFEST="${CORRECTION_MANIFEST:-manifests_master42_n30/master42_correction_manifest.csv}"
SEEDS="${SEEDS:-configs/inference_seeds_master42.json}"
CASES="${CASES:-configs/at_af_7.json}"
RUN_ROOT="${RUN_ROOT:-runs_master42_n30}"
REPORTS="${REPORTS:-reports_master42_n30_ccdfix}"

if [[ "${FORCE:-0}" == "1" || ! -f "$MANIFEST" ]]; then
  "$PYTHON" scripts/build_master42_experiment_manifest.py \
    --seed-file "$SEEDS" \
    --case-config "$CASES" \
    --output "$MANIFEST" \
    --run-root "$RUN_ROOT"
fi

if [[ "${SKIP_GENERATION:-0}" != "1" ]]; then
  : "${AF3_PYTHON:?Set AF3_PYTHON to the AlphaFold 3 environment interpreter}"
  : "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to the AlphaFold 3 model directory/file}"
  "$PYTHON" scripts/run_master42_af3_at_af3_batch.py \
    --manifest "$MANIFEST" \
    --af3-python "$AF3_PYTHON" \
    --model-dir "$AF3_MODEL_DIR" \
    --af3-script "${AF3_ROOT:-$ROOT/alphafold3}/run_alphafold.py" \
    --log-root logs_master42_n30 \
    --max-parallel "${MAX_PARALLEL:-1}" \
    --resume
fi

"$PYTHON" scripts/build_master42_correction_manifest.py \
  --at-output-root "$RUN_ROOT/at_af3" \
  --seed-file "$SEEDS" \
  --target-config-root configs/target_chirality \
  --output "$CORRECTION_MANIFEST" \
  --correction-output-root "$RUN_ROOT/at_af3_chirality_corrected"

"$PYTHON" scripts/run_master42_chirality_correction_batch.py \
  --manifest "$CORRECTION_MANIFEST" \
  --output-root "$RUN_ROOT/at_af3_chirality_corrected" \
  --log-root logs_master42_n30 \
  --max-parallel "${MAX_PARALLEL:-1}" \
  --resume

"$PYTHON" scripts/evaluate_master42_experiment.py \
  --standard-root "$RUN_ROOT/standard_af3" \
  --at-root "$RUN_ROOT/at_af3" \
  --corrected-root "$RUN_ROOT/at_af3_chirality_corrected" \
  --seed-file "$SEEDS" \
  --target-config-root configs/target_chirality \
  --output-root "$REPORTS"

"$PYTHON" scripts/ccdfix_table.py \
  --metrics "$REPORTS/seed_level_metrics.csv" \
  --out "$REPORTS/table_all_d_chirality.md"

echo "All-D pipeline complete: $REPORTS"
