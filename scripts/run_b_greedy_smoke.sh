#!/usr/bin/env bash
# Pipeline B greedy smoke test — n=50, dev split, all 4 variants.
# Runs all variants in one run_pipeline.py invocation (models loaded once each).
# Generator uses num_beams=1 (greedy), matching Pipeline A's do_sample=False.
#
# Usage:
#   nohup bash scripts/run_b_greedy_smoke.sh > results/b_greedy_smoke.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_B=$PROJ/.venv_b/bin/python
PY_A=$PROJ/.venv_a/bin/python
DATA_DIR=/tmp
IMAGES_DIR=/tmp/mmdocrag_data
OUT_DIR=$PROJ/results

echo "=== Pipeline B greedy smoke | n=50 | dev | $(date) ==="
echo "Generator: num_beams=1 (greedy, matching Pipeline A)"

$PY_B $PROJ/scripts/run_pipeline.py \
    --pipeline b \
    --variants original,gold_redundant,negative_redundant,mixed_redundant \
    --split    dev \
    --n        50 \
    --data-dir  "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir   "$OUT_DIR"

echo ""
echo "=== SMOKE TABLE (greedy) ==="
$PY_A $PROJ/scripts/compute_smoke_table.py \
    --results-dir "$OUT_DIR" \
    --data-dir    "$DATA_DIR"

echo ""
echo "=== Done: $(date) ==="
