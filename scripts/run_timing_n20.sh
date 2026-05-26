#!/usr/bin/env bash
# n=20 eval timing trials for Pipeline A and B.
# Measures per-item retrieval and generation times at eval-split resolution.
# Uses original variant only (sufficient to extrapolate × 4 variants).
# Generation params: B uses num_beams=1 (greedy), A uses do_sample=False.
#
# Usage:
#   nohup bash scripts/run_timing_n20.sh \
#     > results/timing_n20/timing_n20.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_A=$PROJ/.venv_a/bin/python
PY_B=$PROJ/.venv_b/bin/python
DATA_DIR=/tmp
IMAGES_DIR=/tmp/mmdocrag_data
OUT_DIR=$PROJ/results/timing_n20

echo "============================================================"
echo "  n=20 eval timing trials — $(date)"
echo "  B: greedy (num_beams=1)  |  A: do_sample=False"
echo "============================================================"

echo ""
echo "--- Trial 1: Pipeline B | original | eval | n=20 | $(date) ---"
$PY_B $PROJ/scripts/run_pipeline.py \
    --pipeline   b \
    --variants   original \
    --split      eval \
    --n          20 \
    --data-dir   "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir    "$OUT_DIR"

echo ""
echo "--- Trial 2: Pipeline A | original | eval | n=20 | $(date) ---"
$PY_A $PROJ/scripts/run_pipeline.py \
    --pipeline   a \
    --variants   original \
    --split      eval \
    --n          20 \
    --data-dir   "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir    "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Both trials complete — $(date)"
echo "============================================================"
