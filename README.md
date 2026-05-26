# MMDocRAG Baseline Study

Citation-noise behavior in multimodal RAG across four dataset variants.
Two pipelines × four variants × two metric tracks.

## One-time setup

### 1. Extract images

```bash
python scripts/extract_images.py \
    --zip-path /path/to/images.zip \
    --out-dir /path/to/data
```

This creates `/path/to/data/images/*.jpg` (~14 826 files). Run once.
Pass `--out-dir` as `--images-dir` in all subsequent commands.

### 2. Create environments

Pipeline A and B have incompatible transformers pins. Use separate venvs.

```bash
python -m venv .venv_a && source .venv_a/bin/activate
pip install -r envs/requirements_a.txt

python -m venv .venv_b && source .venv_b/bin/activate
pip install -r envs/requirements_b.txt
```

### 3. Pre-flight import check

Run in each venv before any pipeline run:

```bash
# In .venv_a
python scripts/preflight.py --pipeline a

# In .venv_b
python scripts/preflight.py --pipeline b
```

If `ColPaliForRetrieval` import fails in venv_a, bump
`transformers>=4.49.0` in `envs/requirements_a.txt` and reinstall.

## Running pipelines

Eight runs total (2 pipelines × 4 variants). Replace `{a|b}` and
`{variant}` accordingly.

```bash
python scripts/run_pipeline.py \
    --pipeline a \
    --variant original \
    --split eval \
    --topk 10 \
    --data-dir /path/to/data \
    --images-dir /path/to/data \
    --out-dir results/
```

Variant choices: `original`, `gold_redundant`, `negative_redundant`,
`mixed_redundant`.

For a 50-question smoke test add `--n 50 --split dev`.

Retrieval scores are cached to `results/{pipeline}_{variant}_retrieval_cache.jsonl`.
Re-running skips retrieval and goes straight to generation.

## Computing metrics

After all 8 runs:

```bash
python scripts/compute_metrics.py --results-dir results/
```

Prints a (pipeline × variant) table for Track 1 (EM/F1/P/R/Jaccard)
and Track 2 (P@10/R@10/F1@10/Jaccard@10 + mean gold-set size),
with 95% bootstrap CIs (1 000 resamples).
