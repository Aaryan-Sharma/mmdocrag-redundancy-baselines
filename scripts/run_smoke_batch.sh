#!/usr/bin/env bash
# Run all 8 smoke tests (pipeline × variant, n=50, dev split).
# Pipeline A original is already done; its results are reused.
# All runs use CUDA_VISIBLE_DEVICES=2,3.
#
# Usage:
#   nohup bash scripts/run_smoke_batch.sh > results/smoke_batch.log 2>&1 &

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

run() {
    local pipeline=$1 variant=$2
    local py
    if [ "$pipeline" = "a" ]; then py=$PY_A; else py=$PY_B; fi
    echo ""
    echo "========================================================"
    echo "  Pipeline ${pipeline^^} | $variant | n=50 | $(date)"
    echo "========================================================"
    $py $PROJ/scripts/run_pipeline.py \
        --pipeline "$pipeline" \
        --variant  "$variant" \
        --split    dev \
        --n        50 \
        --data-dir  "$DATA_DIR" \
        --images-dir "$IMAGES_DIR" \
        --out-dir   "$OUT_DIR"
}

# ---------------------------------------------------------------
# Step 1: Pipeline B venv (must exist before we reach B runs)
# ---------------------------------------------------------------
if [ ! -f "$PY_B" ]; then
    echo "Creating .venv_b ..."
    python3 -m venv "$PROJ/.venv_b"
    "$PROJ/.venv_b/bin/pip" install --quiet --upgrade pip
fi
echo "Syncing .venv_b requirements ..."
"$PROJ/.venv_b/bin/pip" install --quiet -r "$PROJ/envs/requirements_b.txt"
echo ".venv_b ready."

# ---------------------------------------------------------------
# Step 2: VisRAG sanity check (fail-fast before any B run)
# ---------------------------------------------------------------
echo ""
echo "=== VisRAG-Ret sanity check ==="
$PY_B $PROJ/scripts/visrag_sanity.py \
    --images-dir "$IMAGES_DIR" \
    --data-dir   "$DATA_DIR"
echo "VisRAG sanity: PASS"

# ---------------------------------------------------------------
# Step 3: Pipeline A — remaining 3 variants
#         (original already done; its cache+results exist)
# ---------------------------------------------------------------
for v in gold_redundant negative_redundant mixed_redundant; do
    run a "$v"
done

# ---------------------------------------------------------------
# Step 4: Pipeline B — all 4 variants
# ---------------------------------------------------------------
for v in $VARIANTS; do
    run b "$v"
done

# ---------------------------------------------------------------
# Step 5: Smoke table
# ---------------------------------------------------------------
echo ""
echo "=== SMOKE TEST TABLE ==="
$PY_A $PROJ/scripts/compute_smoke_table.py \
    --results-dir "$OUT_DIR" \
    --data-dir    "$DATA_DIR"

echo ""
echo "All smoke tests complete: $(date)"
