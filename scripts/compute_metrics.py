"""
Aggregate metrics across all 8 results files and print a summary table.

Reads:  results/{pipeline}_{variant}.jsonl  (one per pipeline × variant run)
Prints: Track 1 (EM/F1/P/R/Jaccard) and Track 2 (P@10/R@10/F1@10/Jaccard@10
        + mean gold-set size) per run, with 95% bootstrap CIs.

Usage:
    python scripts/compute_metrics.py --results-dir results/
    python scripts/compute_metrics.py --results-dir results/ --csv metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bootstrap import bootstrap_ci
from metrics_answer import compute_answer_metrics
from metrics_retrieval import aggregate_retrieval, compute_retrieval_metrics

PIPELINES = ["a", "b"]
VARIANTS = ["original", "gold_redundant", "negative_redundant", "mixed_redundant"]

ANSWER_METRICS = ["EM", "F1", "Precision", "Recall", "Jaccard"]
RETRIEVAL_METRICS = ["P@10", "R@10", "F1@10", "Jaccard@10"]


def _normalize_tokens(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return set(s.split())


def load_results(path: Path) -> list[dict]:
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_run_metrics(rows: list[dict]) -> dict:
    """
    Compute Track 1 and Track 2 metrics for one (pipeline, variant) run.

    Returns a flat dict with:
        {metric}_mean, {metric}_ci_low, {metric}_ci_high
    for every metric in both tracks, plus mean_gold_set_size and n_skipped.
    """
    t1_scores: dict[str, list[float]] = {m: [] for m in ANSWER_METRICS}
    t2_per_q: list[dict | None] = []
    gold_sizes: list[int] = []

    for row in rows:
        # Track 1
        am = compute_answer_metrics(row["prediction"], row["answer_ref"])
        for m in ANSWER_METRICS:
            t1_scores[m].append(float(am[m]))

        # Track 2
        gold_set = set(row["gold_set"])
        gold_sizes.append(len(gold_set))
        rm = compute_retrieval_metrics(row["retrieved_ids"], gold_set)
        t2_per_q.append(rm)

    result: dict = {}

    # Track 1 bootstrap
    for m in ANSWER_METRICS:
        ci = bootstrap_ci(t1_scores[m])
        result[f"{m}_mean"] = ci["mean"]
        result[f"{m}_ci_low"] = ci["ci_low"]
        result[f"{m}_ci_high"] = ci["ci_high"]

    # Track 2 aggregate + bootstrap
    agg = aggregate_retrieval(t2_per_q, gold_sizes)
    result["mean_gold_set_size"] = agg["mean_gold_set_size"]
    result["n_skipped"] = agg["n_skipped"]
    result["n_questions"] = agg["n_questions"]

    for m in RETRIEVAL_METRICS:
        values = [r[m] for r in t2_per_q if r is not None]
        ci = bootstrap_ci(values)
        result[f"{m}_mean"] = ci["mean"]
        result[f"{m}_ci_low"] = ci["ci_low"]
        result[f"{m}_ci_high"] = ci["ci_high"]

    # Ref-token coverage: fraction of predictions whose token set is a
    # superset of the reference token set (ignoring punctuation/case).
    # Empty references count as covered (convention); flagged separately.
    cov_vals: list[float] = []
    empty_ref_n = 0
    for row in rows:
        ref_toks = _normalize_tokens(row["answer_ref"])
        if not ref_toks:
            empty_ref_n += 1
            cov_vals.append(1.0)
        else:
            pred_toks = _normalize_tokens(row["prediction"])
            cov_vals.append(1.0 if ref_toks.issubset(pred_toks) else 0.0)
    ci = bootstrap_ci(cov_vals)
    result["ref_tok_cov_mean"]    = ci["mean"]
    result["ref_tok_cov_ci_low"]  = ci["ci_low"]
    result["ref_tok_cov_ci_high"] = ci["ci_high"]
    result["ref_tok_cov_empty_ref_n"] = empty_ref_n

    return result


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def print_table(all_results: dict[tuple[str, str], dict]) -> None:
    header = f"{'Pipeline':<10} {'Variant':<22}"
    for m in ANSWER_METRICS + RETRIEVAL_METRICS:
        header += f"  {m:<28}"
    header += f"  {'GoldSetSz':>9}  {'Skipped':>7}"
    print(header)
    print("-" * len(header))

    for pipeline in PIPELINES:
        for variant in VARIANTS:
            key = (pipeline, variant)
            if key not in all_results:
                print(f"{pipeline:<10} {variant:<22}  (no results file)")
                continue
            r = all_results[key]
            row = f"{pipeline:<10} {variant:<22}"
            for m in ANSWER_METRICS:
                row += f"  {fmt(r[f'{m}_mean'], r[f'{m}_ci_low'], r[f'{m}_ci_high']):<28}"
            for m in RETRIEVAL_METRICS:
                row += f"  {fmt(r[f'{m}_mean'], r[f'{m}_ci_low'], r[f'{m}_ci_high']):<28}"
            row += f"  {r['mean_gold_set_size']:>9.2f}  {r['n_skipped']:>7}"
            cov = fmt(r["ref_tok_cov_mean"], r["ref_tok_cov_ci_low"], r["ref_tok_cov_ci_high"])
            row += f"  {cov:<28}"
            print(row)


def write_csv(all_results: dict[tuple[str, str], dict], csv_path: Path) -> None:
    columns = ["pipeline", "variant", "n_questions", "n_skipped", "mean_gold_set_size"]
    for m in ANSWER_METRICS + RETRIEVAL_METRICS:
        columns += [f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high"]
    columns += ["ref_tok_cov_mean", "ref_tok_cov_ci_low", "ref_tok_cov_ci_high",
                "ref_tok_cov_empty_ref_n"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for (pipeline, variant), r in all_results.items():
            row_dict = {"pipeline": pipeline, "variant": variant}
            row_dict.update(r)
            writer.writerow({k: row_dict.get(k, "") for k in columns})
    print(f"CSV written to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to write a CSV summary")
    args = parser.parse_args()

    all_results: dict[tuple[str, str], dict] = {}
    missing: list[str] = []

    for pipeline in PIPELINES:
        for variant in VARIANTS:
            path = args.results_dir / f"{pipeline}_{variant}.jsonl"
            if not path.exists():
                missing.append(str(path))
                continue
            print(f"Loading {path} ...")
            rows = load_results(path)
            all_results[(pipeline, variant)] = compute_run_metrics(rows)

    if missing:
        print(f"\nMissing result files ({len(missing)}):")
        for p in missing:
            print(f"  {p}")

    print("\n=== TRACK 1: Answer Quality | TRACK 2: Evidence Selection ===\n")
    print_table(all_results)

    if args.csv:
        write_csv(all_results, args.csv)


if __name__ == "__main__":
    main()
