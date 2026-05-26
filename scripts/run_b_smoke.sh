#!/usr/bin/env bash
# Run Pipeline B x 4 variants (n=50, dev split), then print smoke table.
# Pipeline A x 4 results are already complete and will be read by the table.
#
# Usage:
#   nohup bash scripts/run_b_smoke.sh > results/b_smoke.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_A=$PROJ/.venv_a/bin/python
PY_B=$PROJ/.venv_b/bin/python
DATA_DIR=/tmp
IMAGES_DIR=/tmp/mmdocrag_data
OUT_DIR=$PROJ/results

VARIANTS="original gold_redundant negative_redundant mixed_redundant"

for v in $VARIANTS; do
    echo ""
    echo "========================================================"
    echo "  Pipeline B | $v | n=50 | $(date)"
    echo "========================================================"
    $PY_B $PROJ/scripts/run_pipeline.py \
        --pipeline b \
        --variant  "$v" \
        --split    dev \
        --n        50 \
        --data-dir  "$DATA_DIR" \
        --images-dir "$IMAGES_DIR" \
        --out-dir   "$OUT_DIR"
done

echo ""
echo "=== SMOKE TEST TABLE ==="
$PY_A $PROJ/scripts/compute_smoke_table.py \
    --results-dir "$OUT_DIR" \
    --data-dir    "$DATA_DIR"

echo ""
echo "All B smoke tests complete: $(date)"
