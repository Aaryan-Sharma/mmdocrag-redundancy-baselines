#!/usr/bin/env bash
# Full eval — Pipeline C (MM-R5 + BGE retriever, Qwen2.5-VL-7B generator).
# All 4 variants, evaluation_20.jsonl (2000 items), top-k=10, greedy decoding.
# Writes to results/full_eval/c/.
#
# Cache-first launch strategy (recommended for wall-clock efficiency):
#   Step 1 — run original on GPU 2 to populate the shared score cache
#             (MM-R5 is run once per question; subsequent variants reuse it):
#     CUDA_VISIBLE_DEVICES=2 bash scripts/run_full_eval_c.sh original
#
#   Step 2 — after Step 1 completes, launch remaining 3 variants in parallel:
#     CUDA_VISIBLE_DEVICES=3 bash scripts/run_full_eval_c.sh gold_redundant     &
#     CUDA_VISIBLE_DEVICES=4 bash scripts/run_full_eval_c.sh negative_redundant &
#     CUDA_VISIBLE_DEVICES=5 bash scripts/run_full_eval_c.sh mixed_redundant    &
#
# Sequential usage (all 4 variants on GPU 2):
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_full_eval_c.sh
#   # or:
#   nohup bash scripts/run_full_eval_c.sh > results/full_eval/c.log 2>&1 &

set -euo pipefail

# Accept an optional single-variant argument for the cache-first strategy.
VARIANT_ARG="${1:-}"
if [ -n "$VARIANT_ARG" ]; then
    VARIANTS_FLAG="$VARIANT_ARG"
    LABEL="variant=$VARIANT_ARG"
else
    VARIANTS_FLAG="original,gold_redundant,negative_redundant,mixed_redundant"
    LABEL="all 4 variants"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_A=$PROJ/.venv_a/bin/python
DATA_DIR=$PROJ/mmdocrag_redundancy_v1
IMAGES_DIR=$PROJ/mmdocrag_redundancy_v1
OUT_C=$PROJ/results/full_eval/c

echo "============================================================"
echo "  Pipeline C full eval — $(date)"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  $LABEL | Split: eval | n=2000 | top-k=10"
echo "  Retriever: MM-R5 (i2vec/MM-R5) + BGE-large-en-v1.5"
echo "  Generator: Qwen2.5-VL-7B-Instruct (do_sample=False)"
echo "============================================================"

# Disk check (monitors results/full_eval/ directory growth)
du_gb() { du -s --block-size=1G "$PROJ/results/full_eval/" 2>/dev/null | cut -f1 | tr -d 'G' || echo "0"; }
DISK_START=$(du_gb)
echo "[disk] results/full_eval/ at start: ${DISK_START}GB"

# Background monitor — warns at 50GB, kills at 100GB
(
  while true; do
    sleep 1800
    USED=$(du_gb)
    FREE=$(df "$PROJ" --output=avail -BG | tail -1 | tr -d 'G ')
    echo "[disk-monitor] $(date +%H:%M:%S) full_eval/ size: ${USED}GB  free: ${FREE}GB"
    if [ "$USED" -gt 100 ]; then
      echo "[disk-monitor] CRITICAL: full_eval/ >100GB. Sending SIGTERM."
      kill -TERM $$ 2>/dev/null || true
    elif [ "$USED" -gt 50 ]; then
      echo "[disk-monitor] WARNING: full_eval/ >50GB."
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null; exit" INT TERM EXIT

echo ""
echo "--- Pipeline C | $LABEL | eval | n=2000 | $(date) ---"
$PY_A "$PROJ/scripts/run_pipeline.py" \
    --pipeline   c \
    --variants   "$VARIANTS_FLAG" \
    --split      eval \
    --data-dir   "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir    "$OUT_C" \
    --topk       10

echo ""
echo "--- Pipeline C generation complete: $(date) ---"

# Per-variant summary
echo ""
echo "=== PIPELINE C VARIANT SUMMARY ==="
PROJ="$PROJ" $PY_A - << 'PYEOF'
import json, os, sys, time
from pathlib import Path
proj = Path(os.environ["PROJ"])
sys.path.insert(0, str(proj / "src"))
from metrics_answer import compute_answer_metrics
from metrics_retrieval import aggregate_retrieval, compute_retrieval_metrics
from bootstrap import bootstrap_ci

OUT = proj / "results/full_eval/c"
VARIANTS = ["original", "gold_redundant", "negative_redundant", "mixed_redundant"]
ts = time.strftime("%Y-%m-%dT%H:%M:%S")

for v in VARIANTS:
    p = OUT / f"c_{v}.jsonl"
    if not p.exists():
        print(f"{ts} | C | {v} | MISSING")
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    n = len(rows)
    n_skip = sum(1 for r in rows if not r.get("prediction", "").strip())
    ems, f1s, t2, gsizes = [], [], [], []
    for r in rows:
        m = compute_answer_metrics(r["prediction"], r["answer_ref"])
        ems.append(float(m["EM"])); f1s.append(float(m["F1"]))
        gs = set(r["gold_set"]); gsizes.append(len(gs))
        t2.append(compute_retrieval_metrics(r["retrieved_ids"], gs))
    agg = aggregate_retrieval(t2, gsizes)
    r10 = bootstrap_ci([x["R@10"] for x in t2 if x])["mean"]
    f10 = bootstrap_ci([x["F1@10"] for x in t2 if x])["mean"]
    em  = bootstrap_ci(ems)["mean"]
    f1  = bootstrap_ci(f1s)["mean"]
    print(f"{ts} | C | {v:25s} | n={n} | skip={n_skip} | R@10={r10:.4f} | F1@10={f10:.4f} | EM={em:.4f} | F1={f1:.4f}")
PYEOF

echo "[disk] results/full_eval/ at C completion: $(du_gb)GB"

echo ""
echo "============================================================"
echo "  Pipeline C DONE — $(date)"
echo "============================================================"
