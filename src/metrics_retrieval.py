"""
Track 2 — Evidence selection metrics.

Measures how well the retriever's top-10 covers the gold quote set,
using set operations on quote_ids.  The gold_set is the expanded set
produced by build_variant() — it includes text-twin ids when applicable.
"""


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_set: set[str],
) -> dict | None:
    """
    Return P@10 / R@10 / F1@10 / Jaccard@10, or None if gold_set is empty.

    retrieved_ids must contain exactly 10 ids (top-10 from re-ranker).
    Caller is responsible for logging None returns as skipped questions.

    Formulas
    --------
    P@10     = |retrieved ∩ gold| / 10
    R@10     = |retrieved ∩ gold| / |gold|
    F1@10    = harmonic mean of P@10 and R@10
    Jaccard  = |retrieved ∩ gold| / |retrieved ∪ gold|
    """
    if not gold_set:
        return None

    retrieved_set = set(retrieved_ids)
    n_intersection = len(retrieved_set & gold_set)
    n_union = len(retrieved_set | gold_set)
    n_gold = len(gold_set)

    precision = n_intersection / 10  # denominator always 10
    recall = n_intersection / n_gold
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = n_intersection / n_union if n_union > 0 else 0.0

    return {"P@10": precision, "R@10": recall, "F1@10": f1, "Jaccard@10": jaccard}


def aggregate_retrieval(
    per_q_results: list[dict | None],
    gold_set_sizes: list[int],
) -> dict:
    """
    Aggregate per-question retrieval dicts into dataset-level means.

    Returns a dict with:
        n_questions      — number of non-skipped questions
        n_skipped        — questions where gold_set was empty
        mean_gold_set_size — mean over ALL questions (needed for cross-
                             variant interpretation since gold_set grows
                             under redundancy variants)
        mean_{metric}    — mean of each metric over non-skipped questions
    """
    valid = [r for r in per_q_results if r is not None]
    n_skipped = len(per_q_results) - len(valid)

    out: dict = {
        "n_questions": len(valid),
        "n_skipped": n_skipped,
        "mean_gold_set_size": (
            sum(gold_set_sizes) / len(gold_set_sizes) if gold_set_sizes else float("nan")
        ),
    }

    if not valid:
        return out

    for metric in ("P@10", "R@10", "F1@10", "Jaccard@10"):
        out[f"mean_{metric}"] = sum(r[metric] for r in valid) / len(valid)

    return out
