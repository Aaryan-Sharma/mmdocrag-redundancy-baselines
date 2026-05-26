# MMDocRAG Redundancy Baseline

Baseline study measuring how citation redundancy affects multimodal document RAG. Two retrieval-generation pipelines (Pipeline A: ColPali+BGE / Qwen2.5-VL-7B; Pipeline B: VisRAG-Ret / MiniCPM-V-2_6) are evaluated across four dataset variants that inject gold, negative, or mixed redundant candidates into the retrieval pool, at full scale (n=2000, eval split, top-k=10).

## Setup

### Requirements

- Python 3.10+
- CUDA GPU (tested on A100 80GB; ~24 GB VRAM minimum per pipeline)
- ~60 GB disk for model weights (cached to `HF_HOME`)
- ~500 MB disk for result JSONL files

Pipelines A and B have incompatible `transformers` version pins and must use **separate virtual environments**.

### 1. Create environments

```bash
python -m venv .venv_a && source .venv_a/bin/activate
pip install -r envs/requirements_a.txt
deactivate

python -m venv .venv_b && source .venv_b/bin/activate
pip install -r envs/requirements_b.txt
deactivate
```

Key pins: Pipeline A requires `transformers>=4.49.0` (Qwen2.5-VL + ColPali via colpali-engine). Pipeline B requires `transformers==4.40.2` (hard pin for VisRAG-Ret and MiniCPM-V-2_6 `trust_remote_code` compatibility).

### 2. Download the frozen dataset

The evaluation dataset (v1, ~315 MB) is hosted externally:

> **Download URL:** `<PLACEHOLDER — fill in before sharing>`

Unpack into `mmdocrag_redundancy_v1/data/`:

```
mmdocrag_redundancy_v1/data/
    original_eval.jsonl
    gold_redundant_eval.jsonl
    negative_redundant_eval.jsonl
    mixed_redundant_eval.jsonl
    dev/
        original_dev.jsonl
        gold_redundant_dev.jsonl
        negative_redundant_dev.jsonl
        mixed_redundant_dev.jsonl
```

Verify checksums after download:

```bash
sha256sum -c mmdocrag_redundancy_v1/checksums.txt
```

### 3. Extract images

```bash
.venv_a/bin/python scripts/extract_images.py \
    --zip-path /path/to/mmdocrag_images.zip \
    --out-dir  /path/to/images
```

This produces ~14 826 JPEG files. Pass the same directory as `--images-dir` in all pipeline runs.

### 4. Pre-flight check

```bash
.venv_a/bin/python scripts/preflight.py --pipeline a
.venv_b/bin/python scripts/preflight.py --pipeline b
```

Both should print `OK` for all imports. If `ColPaliForRetrieval` fails in venv_a, verify `colpali-engine>=0.3.0` is installed.

---

## Reproducing Results

Model weights are downloaded automatically from HuggingFace on first run (requires internet; subsequent runs use the local cache).

Set `HF_HOME` to point at a disk with sufficient space before running:

```bash
export HF_HOME=/path/to/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

### Single-variant run

Replace `PIPELINE` with `a` or `b`, `VARIANT` with one of `original`, `gold_redundant`, `negative_redundant`, `mixed_redundant`:

```bash
# Pipeline A (GPU 0, eval split, 2000 items)
CUDA_VISIBLE_DEVICES=0 .venv_a/bin/python scripts/run_pipeline.py \
    --pipeline   a \
    --variants   VARIANT \
    --split      eval \
    --data-dir   mmdocrag_redundancy_v1/data \
    --images-dir /path/to/images \
    --out-dir    results/full_eval/a \
    --topk       10

# Pipeline B (GPU 0, eval split, 2000 items)
CUDA_VISIBLE_DEVICES=0 .venv_b/bin/python scripts/run_pipeline.py \
    --pipeline   b \
    --variants   VARIANT \
    --split      eval \
    --data-dir   mmdocrag_redundancy_v1/data \
    --images-dir /path/to/images \
    --out-dir    results/full_eval/b \
    --topk       10
```

### All 4 variants in one invocation (models loaded once)

```bash
CUDA_VISIBLE_DEVICES=0 .venv_a/bin/python scripts/run_pipeline.py \
    --pipeline   a \
    --variants   original,gold_redundant,negative_redundant,mixed_redundant \
    --split      eval \
    --data-dir   mmdocrag_redundancy_v1/data \
    --images-dir /path/to/images \
    --out-dir    results/full_eval/a \
    --topk       10
```

Retrieval scores are cached to `results/full_eval/{pipeline}/{pipeline}_{split}_scores.jsonl` after the first variant runs; subsequent variants reuse them (retrieval runs only once per split).

### Compute metrics

After all 8 result files exist:

```bash
.venv_a/bin/python scripts/compute_metrics.py \
    --results-dir results/full_eval \
    --csv         results/full_eval/final_metrics.csv
```

---

## Repo Structure

```
mmdocrag_baseline/
├── envs/
│   ├── requirements_a.txt          # Pipeline A dependencies
│   └── requirements_b.txt          # Pipeline B dependencies
│
├── src/
│   ├── data_loader.py              # Dataset loading, variant construction
│   ├── metrics_answer.py           # Track 1: EM, F1, Precision, Recall, Jaccard
│   ├── metrics_retrieval.py        # Track 2: P@k, R@k, F1@k, Jaccard@k
│   ├── bootstrap.py                # 95% bootstrap CI (1000 resamples)
│   ├── pipeline_a/
│   │   ├── retriever.py            # ColPali v1.3-hf (MaxSim) + BGE-large hybrid
│   │   └── generator.py            # Qwen2.5-VL-7B-Instruct (do_sample=False)
│   └── pipeline_b/
│       ├── retriever.py            # VisRAG-Ret (vision-language dense retrieval)
│       └── generator.py            # MiniCPM-V-2_6 (num_beams=1, greedy)
│
├── scripts/
│   ├── run_pipeline.py             # Main entry point — retrieval + generation
│   ├── compute_metrics.py          # Final metrics table from result JSONLs
│   ├── compute_smoke_table.py      # Quick n=50 smoke summary
│   ├── run_full_eval_a.sh          # Full eval launcher: Pipeline A, GPU 1
│   ├── run_full_eval_b.sh          # Full eval launcher: Pipeline B, GPU 3
│   ├── run_timing_n20.sh           # n=20 timing trials (both pipelines)
│   ├── run_b_greedy_smoke.sh       # Pipeline B greedy smoke test (n=50)
│   ├── preflight.py                # Import sanity checks per pipeline
│   └── extract_images.py           # Unpack images.zip to images/
│
├── tests/
│   ├── test_data_loader.py         # Unit tests: data_loader module
│   └── test_metrics.py             # Unit tests: metrics modules
│
├── mmdocrag_redundancy_v1/
│   ├── README.md                   # Dataset card and variant construction notes
│   ├── checksums.txt               # SHA-256 for all 8 dataset files
│   ├── build_variant.py            # Frozen build script (snapshot of data_loader)
│   └── data/                       # NOT in repo — download separately (see above)
│
├── results/full_eval/
│   ├── a/                          # Pipeline A: 4 × 2000-row result JSONLs
│   ├── b/                          # Pipeline B: 4 × 2000-row result JSONLs
│   ├── final_metrics.csv           # All 8 runs: Track 1 + Track 2 with 95% CIs
│   ├── final_metrics_table.txt     # Human-readable table version
│   ├── REPORT.md                   # Full results write-up with analysis
│   ├── qualitative_examples.md     # 20 representative prediction examples
│   └── qualitative/
│       ├── 01_when_redundancy_happens.md
│       ├── 02_answer_changes.md
│       └── 03_retrieval_diffs.md
│
├── .gitignore
├── README.md                       # Quick-start (original)
└── REPO_README.md                  # This file — full reproduction guide
```

---

## Model Checkpoints

All weights are downloaded automatically via `transformers` / `sentence-transformers` / `colpali-engine` on first run. No manual download step is needed beyond setting `HF_HOME`.

| Component | HuggingFace ID | Size (approx) |
|---|---|---|
| Pipeline A — retriever (image) | `vidore/colpali-v1.3-hf` | ~10 GB |
| Pipeline A — retriever (text) | `BAAI/bge-large-en-v1.5` | ~1.3 GB |
| Pipeline A — generator | `Qwen/Qwen2.5-VL-7B-Instruct` | ~15 GB |
| Pipeline B — retriever | `openbmb/VisRAG-Ret` | ~7 GB |
| Pipeline B — generator | `openbmb/MiniCPM-V-2_6` | ~16 GB |

---

## Citation / Acknowledgments

> **[PLACEHOLDER]** Add dataset citation, paper reference, and any acknowledgments here before making the repo public.

This project uses:
- [ColPali](https://github.com/illuin-tech/colpali) (vidore/colpali-v1.3-hf)
- [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (BAAI)
- [VisRAG-Ret](https://huggingface.co/openbmb/VisRAG-Ret) (OpenBMB)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Qwen)
- [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) (OpenBMB)
