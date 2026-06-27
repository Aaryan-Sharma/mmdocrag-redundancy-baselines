#!/usr/bin/env bash
# Smoke test — Pipeline C (MM-R5 + BGE retriever, Qwen2.5-VL-7B generator), GPU 2.
# n=50, dev split, original variant only.
#
# Usage (run inside tmux session smoke_c):
#   tmux new-session -d -s smoke_c
#   tmux send-keys -t smoke_c \
#     "bash scripts/run_smoke_c.sh 2>&1 | tee results/smoke_c/smoke_c.log" Enter

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HF_HOME="${HF_HOME:-$PROJ/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
PY_A=$PROJ/.venv_a/bin/python
DATA_DIR=$PROJ/mmdocrag_redundancy_v1
IMAGES_DIR=$PROJ/mmdocrag_redundancy_v1
OUT_C=$PROJ/results/smoke_c

echo "============================================================"
echo "  Pipeline C smoke test — $(date)"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Variant: original | Split: dev | n=50 | top-k=10"
echo "  Retriever: MM-R5 (i2vec/MM-R5) + BGE-large-en-v1.5"
echo "  Generator: Qwen2.5-VL-7B-Instruct (greedy, do_sample=False)"
echo "  max_new_tokens: 4096 (retriever CoT) / 128 (generator)"
echo "============================================================"

# Confirm data and image dirs exist before loading any models
if [ ! -f "$DATA_DIR/dev_20.jsonl" ]; then
    echo "ERROR: $DATA_DIR/dev_20.jsonl not found. Check DATA_DIR."
    exit 1
fi
if [ ! -d "$IMAGES_DIR/images" ]; then
    echo "ERROR: $IMAGES_DIR/images not found. Check IMAGES_DIR."
    exit 1
fi

mkdir -p "$OUT_C"

$PY_A "$PROJ/scripts/run_pipeline.py" \
    --pipeline   c \
    --variants   original \
    --split      dev \
    --n          50 \
    --data-dir   "$DATA_DIR" \
    --images-dir "$IMAGES_DIR" \
    --out-dir    "$OUT_C" \
    --topk       10

echo ""
echo "--- Smoke retrieval + generation complete: $(date) ---"
echo ""

# Quick metrics summary
echo "=== Pipeline C smoke metrics ==="
PROJ="$PROJ" $PY_A - << 'PYEOF'
import json, os, sys
from pathlib import Path
proj = Path(os.environ["PROJ"])
sys.path.insert(0, str(proj / "src"))
from metrics_answer import compute_answer_metrics
from metrics_retrieval import compute_retrieval_metrics
from bootstrap import bootstrap_ci

p = proj / "results/smoke_c/c_original.jsonl"
if not p.exists():
    print("Result file not found:", p)
    sys.exit(1)

rows = [json.loads(l) for l in open(p) if l.strip()]
n = len(rows)
ems, f1s, t2, gsizes = [], [], [], []
parse_ok = 0
for r in rows:
    m = compute_answer_metrics(r["prediction"], r["answer_ref"])
    ems.append(float(m["EM"])); f1s.append(float(m["F1"]))
    gs = set(r["gold_set"]); gsizes.append(len(gs))
    t2.append(compute_retrieval_metrics(r["retrieved_ids"], gs))
    if r.get("prediction", "").strip():
        parse_ok += 1

r10 = bootstrap_ci([x["R@10"] for x in t2])
f10 = bootstrap_ci([x["F1@10"] for x in t2])
em  = bootstrap_ci(ems)
f1  = bootstrap_ci(f1s)

print(f"n={n}  |  parse_ok={parse_ok}/{n}")
print(f"R@10  = {r10['mean']:.4f}  [{r10['ci_low']:.4f}, {r10['ci_high']:.4f}]")
print(f"F1@10 = {f10['mean']:.4f}  [{f10['ci_low']:.4f}, {f10['ci_high']:.4f}]")
print(f"EM    = {em['mean']:.4f}   [{em['ci_low']:.4f}, {em['ci_high']:.4f}]")
print(f"F1    = {f1['mean']:.4f}   [{f1['ci_low']:.4f}, {f1['ci_high']:.4f}]")

# Modality mix in top-10
import statistics
img_counts, txt_counts = [], []
for r in rows:
    top10 = r["retrieved_ids"][:10]
    # Can't easily resolve type without loading data_loader; report id counts only
    img_counts.append(len(top10))
print(f"\ntop-10 ids per question: min={min(img_counts)} max={max(img_counts)} mean={statistics.mean(img_counts):.1f}")
PYEOF

echo ""
echo "--- Results in: $OUT_C ---"
echo "============================================================"
echo "  Pipeline C smoke DONE — $(date)"
echo "============================================================"
