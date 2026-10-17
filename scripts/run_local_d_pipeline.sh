#!/usr/bin/env bash
# Reproduce the post-cutoff Local-D arm (9 targets x 30 matched seeds).
#
# Required environment:
#   AF3_PYTHON / AF3_MODEL_DIR   AlphaFold 3 environment and model path
#   AF3_ROOT                     checkout containing run_alphafold.py
#
# The Local-D inputs are built from the curated manifests in
# data/local_d/local_d_curated_v1_1. The nine post-cutoff targets are never used
# to fit or select anything.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

: "${AF3_PYTHON:?Set AF3_PYTHON to the AlphaFold 3 environment interpreter}"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to the AlphaFold 3 model directory/file}"
export AF3_PYTHON AF3_MODEL_DIR
export AF3_ROOT="${AF3_ROOT:-$ROOT/alphafold3}"

"$PYTHON" scripts/local_d_curated_v1_1/prepare_post_cutoff_run.py
"$PYTHON" scripts/local_d_curated_v1_1/run_post_cutoff_linux.py --phase all --resume

echo "Local-D pipeline complete: runs/local_d_curated_v1_1_post_cutoff"
