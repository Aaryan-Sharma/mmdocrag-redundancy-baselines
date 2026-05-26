#!/usr/bin/env bash
# Full eval — Pipeline B (MiniCPM-V-2_6 + VisRAG-Ret), GPU 3.
# All 4 variants, evaluation_20.jsonl (2000 items), top-k=10.
# Generator: num_beams=1 (greedy), matching Pipeline A's do_sample=False.
#
# Usage:
#   nohup bash scripts/run_full_eval_b.sh > results/full_eval/b.log 2>&1 &

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_B=$PROJ/.venv_b/bin/python
PY_A=$PROJ/.venv_a/bin/python
DATA_DIR=/tmp
IMAGES_DIR=/tmp/mmdocrag_data
OUT_B=$PROJ/results/full_eval/b

echo "============================================================"
echo "  Pipeline B full eval — $(date)"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Variants: all 4 | Split: eval | n=2000 | top-k=10"
echo "  Generator: MiniCPM-V-2_6, sampling=False, num_beams=1 (greedy)"
echo "============================================================"

# Disk check (monitors results/full_eval/ directory growth, not total filesystem)
du_gb() { du -s --block-size=1G "$PROJ/results/full_eval/" 2>/dev/null | cut -f1 | tr -d 'G' || echo "0"; }
DISK_START=$(du_gb)
echo "[disk] results/full_eval/ at start: ${DISK_START}GB"

# Background monitor — checks every 30 min, warns at 50GB, kills at 100GB
(
  while true; do
    sleep 1800
    USED=$(du_gb)
    FREE=$(df "$PROJ" --output=avail -BG | tail -1 | tr -d 'G ')
    echo "[disk-monitor] $(date +%H:%M:%S) full_eval/ size: ${USED}GB  free: ${FREE}GB"
    if [ "$USED" -gt 100 ]; then
      echo "[disk-monitor] CRITICAL: full_eval/ >100GB. Sending SIGTERM to pipeline."
      kill -TERM $$ 2>/dev/null || true
    elif [ "$USED" -gt 50 ]; then
      echo "[disk-monitor] WARNING: full_eval/ >50GB."
    fi
  done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null; exit" INT TERM EXIT

echo ""
echo "--- Pipeline B | all variants | eval | n=2000 | $(date) ---"
$PY_B $PROJ/scripts/run_pipeline.py \
    --pipeline   b \
    --variants   original,gold_redundant,negative_redundant,mixed_redundant \
    --split      eval \
    --data-dir   "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir    "$OUT_B" \
    --topk       10

echo ""
echo "--- Pipeline B generation complete: $(date) ---"

# Per-variant summary lines
echo ""
echo "=== PIPELINE B VARIANT SUMMARY ==="
PROJ="$PROJ" $PY_A - << 'PYEOF'
import json, os, sys, time
from pathlib import Path
proj = Path(os.environ["PROJ"])
sys.path.insert(0, str(proj / "src"))
from metrics_answer import compute_answer_metrics
from metrics_retrieval import aggregate_retrieval, compute_retrieval_metrics
from bootstrap import bootstrap_ci

OUT = proj / "results/full_eval/b"
VARIANTS = ["original", "gold_redundant", "negative_redundant", "mixed_redundant"]
ts = time.strftime("%Y-%m-%dT%H:%M:%S")

for v in VARIANTS:
    p = OUT / f"b_{v}.jsonl"
    if not p.exists():
        print(f"{ts} | B | {v} | MISSING")
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    n = len(rows)
    n_skip = sum(1 for r in rows if not r.get("prediction","").strip())
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
    print(f"{ts} | B | {v:25s} | n={n} | skip={n_skip} | R@10={r10:.4f} | F1@10={f10:.4f} | EM={em:.4f} | F1={f1:.4f}")
PYEOF

# Disk check at B completion
echo "[disk] results/full_eval/ at B completion: $(du_gb)GB"

echo ""
echo "============================================================"
echo "  Pipeline B DONE — $(date)"
echo "  (Pipeline A still running — final metrics computed in a.log)"
echo "============================================================"
