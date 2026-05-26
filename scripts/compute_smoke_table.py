"""
Compute the 8-run smoke-test summary table.

Reads results/{pipeline}_{variant}.jsonl (n=50 dev runs) and
dev_20.jsonl to compute pool sizes via build_variant.

Output columns:
  pipeline | variant | n_skipped | mean_pool_size | mean_gold_set_size
  | R@10 | F1@10 | EM | F1

Usage:
    python scripts/compute_smoke_table.py \
        --results-dir results/ --data-dir /tmp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bootstrap import bootstrap_ci
from data_loader import build_variant, load_jsonl
from metrics_answer import compute_answer_metrics
from metrics_retrieval import aggregate_retrieval, compute_retrieval_metrics

PIPELINES = ["a", "b"]
VARIANTS = ["original", "gold_redundant", "negative_redundant", "mixed_redundant"]

# expected mean_pool_size per variant (roughly; exact depends on data)
POOL_EXPECT = {
    "original":           "~20",
    "gold_redundant":     "~21",
    "negative_redundant": "~29",
    "mixed_redundant":    "~30",
}


def compute_pool_sizes(items: list[dict], variant: str) -> list[int]:
    sizes = []
    for item in items:
        candidates, _ = build_variant(item, variant)
        sizes.append(len(candidates))
    return sizes


def run_metrics(rows: list[dict]) -> dict:
    t1 = {m: [] for m in ["EM", "F1", "Precision", "Recall", "Jaccard"]}
    t2_per_q: list = []
    gold_sizes: list[int] = []

    for row in rows:
        am = compute_answer_metrics(row["prediction"], row["answer_ref"])
        for m in t1:
            t1[m].append(float(am[m]))
        gold_set = set(row["gold_set"])
        gold_sizes.append(len(gold_set))
        t2_per_q.append(compute_retrieval_metrics(row["retrieved_ids"], gold_set))

    agg = aggregate_retrieval(t2_per_q, gold_sizes)
    result = {
        "n_questions": agg["n_questions"],
        "n_skipped":   agg["n_skipped"],
        "mean_gold_set_size": agg["mean_gold_set_size"],
    }
    for m in ["EM", "F1"]:
        result[f"{m}_mean"] = bootstrap_ci(t1[m])["mean"]
    for m in ["R@10", "F1@10"]:
        vals = [r[m] for r in t2_per_q if r is not None]
        result[f"{m}_mean"] = bootstrap_ci(vals)["mean"]

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    dev_path = args.data_dir / "dev_20.jsonl"
    all_items = load_jsonl(dev_path)[:50]   # smoke tests used first 50

    col_w = [4, 20, 10, 16, 20, 8, 8, 8, 8]
    headers = ["Pipe", "Variant", "n_skipped", "mean_pool_size",
               "mean_gold_set_sz", "R@10", "F1@10", "EM", "F1"]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    any_missing = False
    flags: list[str] = []

    for pipeline in PIPELINES:
        for variant in VARIANTS:
            path = args.results_dir / f"{pipeline}_{variant}.jsonl"
            if not path.exists():
                row_str = "  ".join(
                    [pipeline.upper().ljust(col_w[0]),
                     variant.ljust(col_w[1])]
                ) + "  (missing)"
                print(row_str)
                any_missing = True
                continue

            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh if l.strip()]

            if len(rows) < 50:
                flags.append(f"  WARN: {pipeline}_{variant} has only {len(rows)} rows (expected 50)")

            pool_sizes = compute_pool_sizes(all_items[:len(rows)], variant)
            mean_pool = sum(pool_sizes) / len(pool_sizes)

            m = run_metrics(rows)

            # Sanity flags
            if variant == "original" and not (19.5 < mean_pool < 20.5):
                flags.append(f"  FLAG: {pipeline}_{variant} mean_pool={mean_pool:.1f} (expected ~20)")
            if variant in ("gold_redundant", "mixed_redundant"):
                if m["mean_gold_set_size"] <= 1.6:
                    flags.append(f"  FLAG: {pipeline}_{variant} mean_gold_sz={m['mean_gold_set_size']:.2f} "
                                  "— gold_set not expanding as expected")
            if m["n_skipped"] > 0:
                flags.append(f"  FLAG: {pipeline}_{variant} n_skipped={m['n_skipped']}")

            cells = [
                pipeline.upper(),
                variant,
                str(m["n_skipped"]),
                f"{mean_pool:.1f} ({POOL_EXPECT[variant]})",
                f"{m['mean_gold_set_size']:.2f}",
                f"{m['R@10_mean']:.3f}",
                f"{m['F1@10_mean']:.3f}",
                f"{m['EM_mean']:.3f}",
                f"{m['F1_mean']:.3f}",
            ]
            print("  ".join(c.ljust(w) for c, w in zip(cells, col_w)))

    print("=" * len(header_line))

    if flags:
        print("\nSANITY FLAGS:")
        for f in flags:
            print(f)

    if any_missing:
        print("\nSome result files are missing — re-run when all smoke tests complete.")


if __name__ == "__main__":
    main()
