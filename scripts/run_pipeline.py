"""
Main pipeline runner — multi-variant, score-cache-aware.

Processes one or more (pipeline × variant) combinations in a single
invocation, loading models once per phase:
  Phase 1: load retriever → score all variants → unload retriever
  Phase 2: load generator → generate all variants → unload generator

Score cache (Diagnostic 1)
--------------------------
Raw per-candidate scores are stored variant-agnostically in:
  {out_dir}/{pipeline}_{split}_scores.jsonl

Each row: {"q_id": "...", "scores": {"candidate_id": score, ...}}

This means the 20 base candidates shared across all variants are
scored exactly once.  Each subsequent variant only scores the new
twin candidates it introduces:
  original           →  20 new candidates
  gold_redundant     →  ~1 new (gold-image text twin)
  negative_redundant →  ~7-9 new (non-gold-image text twins)
  mixed_redundant    →   0 new (all twins already scored above)

Output files (all in --out-dir)
--------------------------------
Score cache (variant-agnostic):
  {pipeline}_{split}_scores.jsonl

Retrieval results (per variant):
  {pipeline}_{variant}_{split}[_n{n}]_retrieval_cache.jsonl

Final predictions (per variant):
  {pipeline}_{variant}.jsonl

Usage
-----
# Full eval, all 4 variants, Pipeline B (Diagnostic 3: single invocation):
python scripts/run_pipeline.py \\
    --pipeline b \\
    --variants original,gold_redundant,negative_redundant,mixed_redundant \\
    --split eval \\
    --data-dir /tmp --images-dir /tmp/mmdocrag_data --out-dir results/eval

# Timing trial, n=200:
python scripts/run_pipeline.py \\
    --pipeline b --variants original,gold_redundant,negative_redundant,mixed_redundant \\
    --split eval --n 200 \\
    --data-dir /tmp --images-dir /tmp/mmdocrag_data --out-dir results/timing

# Single-variant smoke test (backward-compatible):
python scripts/run_pipeline.py \\
    --pipeline a --variants original --split dev --n 50 \\
    --data-dir /tmp --images-dir /tmp/mmdocrag_data --out-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import build_variant, load_jsonl, parse_answer

SPLIT_FILES = {
    "eval": "evaluation_20.jsonl",
    "dev":  "dev_20.jsonl",
}

ALL_VARIANTS = [
    "original", "gold_redundant", "negative_redundant", "mixed_redundant"
]


# ---------------------------------------------------------------------------
# Score cache helpers (variant-agnostic)
# ---------------------------------------------------------------------------

def _score_cache_path(out_dir: Path, pipeline: str, split: str) -> Path:
    return out_dir / f"{pipeline}_{split}_scores.jsonl"


def load_score_cache(path: Path) -> dict[str, dict[str, float]]:
    """Load {q_id → {candidate_id → raw_score}} from JSONL score cache."""
    cache: dict[str, dict[str, float]] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                # Last occurrence wins (allows incremental appends)
                cache[row["q_id"]] = row["scores"]
    return cache


def append_score_row(path: Path, q_id: str, scores: dict[str, float]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"q_id": q_id, "scores": scores}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-variant retrieval cache helpers
# ---------------------------------------------------------------------------

def _retrieval_cache_path(
    out_dir: Path, pipeline: str, variant: str, split: str, n: int | None
) -> Path:
    tag = f"{pipeline}_{variant}_{split}"
    if n is not None:
        tag += f"_n{n}"
    return out_dir / f"{tag}_retrieval_cache.jsonl"


def _results_path(out_dir: Path, pipeline: str, variant: str) -> Path:
    return out_dir / f"{pipeline}_{variant}.jsonl"


def _check_retrieval_cache(path: Path, needed_qids: set[str]) -> list[dict] | None:
    """Return ordered rows if cache covers all needed q_ids, else None."""
    if not path.exists():
        return None
    rows = load_jsonl(path)
    cached_qids = {r["q_id"] for r in rows}
    if needed_qids.issubset(cached_qids):
        qid_map = {r["q_id"]: r for r in rows}
        return rows  # preserve original cache order; caller re-orders by item
    print(f"  Cache incomplete ({len(cached_qids)}/{len(needed_qids)} qids): {path.name}")
    return None


# ---------------------------------------------------------------------------
# Evidence loading
# ---------------------------------------------------------------------------

def _load_evidence(
    candidates_by_id: dict[str, dict],
    retrieved_ids: list[str],
    images_dir: Path,
) -> list:
    from PIL import Image

    evidence = []
    for qid in retrieved_ids:
        cand = candidates_by_id[qid]
        if cand["type"] == "image":
            evidence.append(Image.open(images_dir / cand["img_path"]).convert("RGB"))
        else:
            evidence.append(cand["text"])
    return evidence


# ---------------------------------------------------------------------------
# Multi-variant retrieval phase
# ---------------------------------------------------------------------------

def run_retrieval_phase(
    items: list[dict],
    variants: list[str],
    pipeline: str,
    images_dir: Path,
    topk: int,
    out_dir: Path,
    split: str,
    n: int | None,
    score_cache_path: Path,
) -> dict[str, list[dict]]:
    """
    Score all variants using a shared candidate score cache.
    Loads the retriever exactly once regardless of how many variants are
    requested.  Returns {variant: [retrieval_cache_rows]} in item order.
    """
    needed_qids = {item["q_id"] for item in items}

    # Check which variants already have complete retrieval caches
    retrieval_caches: dict[str, list[dict]] = {}
    variants_todo: list[str] = []
    for variant in variants:
        cp = _retrieval_cache_path(out_dir, pipeline, variant, split, n)
        cached = _check_retrieval_cache(cp, needed_qids)
        if cached is not None:
            print(f"  Retrieval cache hit ({variant}): {cp.name}")
            qid_map = {r["q_id"]: r for r in cached}
            retrieval_caches[variant] = [qid_map[item["q_id"]] for item in items]
        else:
            variants_todo.append(variant)

    if not variants_todo:
        return retrieval_caches

    # Load variant-agnostic score cache
    score_cache = load_score_cache(score_cache_path)
    print(f"  Score cache: {len(score_cache)} q_ids pre-loaded from {score_cache_path.name}")

    # Load retriever (once for all variants)
    if pipeline == "a":
        from pipeline_a.retriever import load_models, rerank_with_cache, unload_models
    elif pipeline == "b":
        from pipeline_b.retriever import load_models, rerank_with_cache, unload_models
    else:
        from pipeline_c.retriever import load_models, rerank_with_cache, unload_models
    load_models()

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_files = {}
    for variant in variants_todo:
        cp = _retrieval_cache_path(out_dir, pipeline, variant, split, n)
        cache_files[variant] = open(cp, "w", encoding="utf-8")
        retrieval_caches[variant] = []

    t0 = time.time()
    for i, item in enumerate(items):
        q_id = item["q_id"]
        # Per-question score cache (all previously scored candidates)
        q_cache = score_cache.get(q_id, {}).copy()
        new_q_scores: dict[str, float] = {}
        q_state = None  # carries query embedding across variant calls

        for variant in variants_todo:
            candidates, gold_set = build_variant(item, variant)
            retrieved_ids, new_scores, q_state = rerank_with_cache(
                query=item["question"],
                candidates=candidates,
                images_dir=images_dir,
                topk=topk,
                score_cache=q_cache,
                q_state=q_state,
            )
            q_cache.update(new_scores)
            new_q_scores.update(new_scores)

            row = {
                "q_id": q_id,
                "retrieved_ids": retrieved_ids,
                "gold_set": sorted(gold_set),
                "answer_ref": parse_answer(item["answer_short"]),
            }
            cache_files[variant].write(json.dumps(row, ensure_ascii=False) + "\n")
            retrieval_caches[variant].append(row)

        # Persist any newly computed scores for this question
        if new_q_scores:
            merged = {**score_cache.get(q_id, {}), **new_q_scores}
            score_cache[q_id] = merged
            append_score_row(score_cache_path, q_id, merged)

        if (i + 1) % 100 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate if rate > 0 else 0
            print(f"  Retrieval: {i+1}/{len(items)} "
                  f"({rate:.2f} q/s, ETA {eta/60:.1f} min)")

    for fh in cache_files.values():
        fh.close()
    unload_models()

    elapsed = time.time() - t0
    print(f"  Retrieval done: {len(items)} items × {len(variants_todo)} variants "
          f"in {elapsed/60:.1f} min ({elapsed/len(items):.2f} s/item)")
    return retrieval_caches


# ---------------------------------------------------------------------------
# Multi-variant generation phase
# ---------------------------------------------------------------------------

def run_generation_phase(
    items: list[dict],
    retrieval_caches: dict[str, list[dict]],
    variants: list[str],
    pipeline: str,
    images_dir: Path,
    out_dir: Path,
    max_new_tokens: int,
) -> None:
    """
    Generate answers for all variants; loads the generator exactly once.
    """
    item_map = {item["q_id"]: item for item in items}

    if pipeline == "a":
        from pipeline_a.generator import generate, load_model, unload_model
    elif pipeline == "b":
        from pipeline_b.generator import generate, load_model, unload_model
    else:
        from pipeline_c.generator import generate, load_model, unload_model
    load_model()

    t0 = time.time()
    for variant in variants:
        cache_rows = retrieval_caches[variant]
        out_path = _results_path(out_dir, pipeline, variant)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_failed = 0
        vt0 = time.time()
        with open(out_path, "w", encoding="utf-8") as fh:
            for i, cache_row in enumerate(cache_rows):
                q_id = cache_row["q_id"]
                item = item_map[q_id]
                candidates, _ = build_variant(item, variant)
                candidates_by_id = {c["quote_id"]: c for c in candidates}
                evidence = _load_evidence(
                    candidates_by_id, cache_row["retrieved_ids"], images_dir
                )

                try:
                    prediction = generate(
                        question=item["question"],
                        evidence_items=evidence,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as exc:
                    print(f"  WARN: generation failed q_id={q_id}: {exc}")
                    prediction = ""
                    n_failed += 1

                result = {
                    "q_id": q_id,
                    "retrieved_ids": cache_row["retrieved_ids"],
                    "gold_set": cache_row["gold_set"],
                    "answer_ref": cache_row["answer_ref"],
                    "prediction": prediction,
                }
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")

                if (i + 1) % 100 == 0 or (i + 1) == len(cache_rows):
                    elapsed = time.time() - vt0
                    rate = (i + 1) / elapsed
                    eta = (len(cache_rows) - i - 1) / rate if rate > 0 else 0
                    print(f"    [{variant}] {i+1}/{len(cache_rows)} "
                          f"({rate:.2f} q/s, ETA {eta/60:.1f} min)")

        elapsed_v = time.time() - vt0
        print(f"  {variant}: {len(cache_rows)} generated in {elapsed_v/60:.1f} min"
              + (f" ({n_failed} failed)" if n_failed else "")
              + f" → {out_path.name}")

    unload_model()
    print(f"  Generation done: {(time.time()-t0)/60:.1f} min total")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pipeline × variant(s) with shared model loads and score cache."
    )
    parser.add_argument("--pipeline", choices=["a", "b", "c"], required=True)
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help="Comma-separated variants (default: all 4)",
    )
    parser.add_argument("--split", choices=["eval", "dev"], default="eval")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default="results", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N items (timing trial / smoke test)"
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ALL_VARIANTS:
            sys.exit(f"Unknown variant {v!r}. Valid: {ALL_VARIANTS}")

    jsonl_path = args.data_dir / SPLIT_FILES[args.split]
    if not jsonl_path.exists():
        sys.exit(f"Data file not found: {jsonl_path}")

    items = load_jsonl(jsonl_path)
    if args.n is not None:
        items = items[: args.n]
        print(f"Limiting to first {len(items)} items ({args.split} split).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_cache_path = _score_cache_path(args.out_dir, args.pipeline, args.split)

    t_start = time.time()
    print(f"\n=== Pipeline {args.pipeline.upper()} | {variants} | {args.split} "
          f"| top-{args.topk} | n={len(items)} | {time.strftime('%H:%M:%S')} ===")

    retrieval_caches = run_retrieval_phase(
        items=items,
        variants=variants,
        pipeline=args.pipeline,
        images_dir=args.images_dir,
        topk=args.topk,
        out_dir=args.out_dir,
        split=args.split,
        n=args.n,
        score_cache_path=score_cache_path,
    )

    run_generation_phase(
        items=items,
        retrieval_caches=retrieval_caches,
        variants=variants,
        pipeline=args.pipeline,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
    )

    total_min = (time.time() - t_start) / 60
    print(f"\nTotal wall-clock: {total_min:.1f} min  ({total_min/len(items)*60:.1f} s/item)")
    print(f"Results written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
