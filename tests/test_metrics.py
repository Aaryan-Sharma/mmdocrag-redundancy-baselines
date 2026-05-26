"""
Unit tests for metrics_answer, metrics_retrieval, and bootstrap.
All inputs are hand-crafted with pre-computed golden values.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics_answer import compute_answer_metrics, normalize_answer
from metrics_retrieval import aggregate_retrieval, compute_retrieval_metrics
from bootstrap import bootstrap_ci


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ===========================================================================
# normalize_answer
# ===========================================================================

def test_normalize_lowercase():
    assert normalize_answer("The Cat SAT") == "cat sat"


def test_normalize_strips_articles():
    # only "a", "an", "the" are stripped; "and" is kept
    assert normalize_answer("a cat and the dog an elephant") == "cat and dog elephant"


def test_normalize_strips_punctuation():
    assert normalize_answer("hello, world!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  too   many   spaces  ") == "too many spaces"


def test_normalize_empty():
    assert normalize_answer("") == ""


# ===========================================================================
# compute_answer_metrics — edge cases
# ===========================================================================

def test_both_empty():
    m = compute_answer_metrics("", "")
    assert m == {"EM": 1, "F1": 1.0, "Precision": 1.0, "Recall": 1.0, "Jaccard": 1.0}


def test_pred_empty_ref_nonempty():
    m = compute_answer_metrics("", "answer")
    assert m["EM"] == 0
    assert m["F1"] == 0.0
    assert m["Precision"] == 0.0
    assert m["Recall"] == 0.0
    assert m["Jaccard"] == 0.0


def test_pred_nonempty_ref_empty():
    m = compute_answer_metrics("answer", "")
    assert m["EM"] == 0
    assert m["F1"] == 0.0


def test_exact_match():
    m = compute_answer_metrics("cat sat mat", "cat sat mat")
    assert m["EM"] == 1
    assert close(m["F1"], 1.0)
    assert close(m["Precision"], 1.0)
    assert close(m["Recall"], 1.0)
    assert close(m["Jaccard"], 1.0)


def test_exact_match_after_normalization():
    # articles and punctuation should not prevent EM
    m = compute_answer_metrics("The Answer.", "answer")
    assert m["EM"] == 1
    assert close(m["F1"], 1.0)


def test_partial_overlap():
    # pred="46", ref="46 27 64"
    # pred_tokens={46:1}, ref_tokens={46:1, 27:1, 64:1}
    # intersection=1, n_pred=1, n_ref=3, union=3
    # P=1/1=1, R=1/3, F1=2*(1)*(1/3)/(1+1/3)=0.5, J=1/3
    m = compute_answer_metrics("46", "46 27 64")
    assert close(m["Precision"], 1.0)
    assert close(m["Recall"], 1 / 3)
    assert close(m["F1"], 0.5)
    assert close(m["Jaccard"], 1 / 3)
    assert m["EM"] == 0


def test_no_overlap():
    # pred="dog", ref="cat"
    m = compute_answer_metrics("dog", "cat")
    assert m["EM"] == 0
    assert close(m["F1"], 0.0)
    assert close(m["Precision"], 0.0)
    assert close(m["Recall"], 0.0)
    assert close(m["Jaccard"], 0.0)


def test_repeated_tokens_multiset():
    # pred="a a b", ref="a b b"  (after normalization: "b" ref has "b b")
    # pred_norm="b" (article "a" stripped), ref_norm="b b"
    # pred_counts={b:1}, ref_counts={b:2}
    # intersection (min) = {b:1} -> 1
    # union (max)        = {b:2} -> 2
    # P=1/1=1, R=1/2=0.5, F1=2/3, J=1/2
    m = compute_answer_metrics("a a b", "a b b")
    assert close(m["Precision"], 1.0)
    assert close(m["Recall"], 0.5)
    assert close(m["F1"], 2 / 3)
    assert close(m["Jaccard"], 0.5)


def test_article_stripping_in_metrics():
    # "the answer is" -> "answer is" (2 tokens)
    # "answer"        -> "answer"    (1 token)
    # intersection={answer:1}=1, n_pred=2, n_ref=1, union=2
    # P=1/2, R=1/1=1, F1=2/3, J=1/2
    m = compute_answer_metrics("the answer is", "answer")
    assert close(m["Precision"], 0.5)
    assert close(m["Recall"], 1.0)
    assert close(m["F1"], 2 / 3)
    assert close(m["Jaccard"], 0.5)


# ===========================================================================
# compute_retrieval_metrics
# ===========================================================================

def test_retrieval_empty_gold_returns_none():
    result = compute_retrieval_metrics(["q1"] * 10, set())
    assert result is None


def test_retrieval_full_recall():
    # Both gold quotes are in top-10
    retrieved = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
    gold = {"q1", "q2"}
    m = compute_retrieval_metrics(retrieved, gold)
    # intersection=2, union=10, P=2/10=0.2, R=2/2=1, J=2/10=0.2
    assert close(m["P@10"], 0.2)
    assert close(m["R@10"], 1.0)
    assert close(m["F1@10"], 2 * 0.2 * 1.0 / (0.2 + 1.0))
    assert close(m["Jaccard@10"], 2 / 10)


def test_retrieval_no_overlap():
    retrieved = [f"r{i}" for i in range(10)]
    gold = {"g1", "g2"}
    m = compute_retrieval_metrics(retrieved, gold)
    assert close(m["P@10"], 0.0)
    assert close(m["R@10"], 0.0)
    assert close(m["F1@10"], 0.0)
    assert close(m["Jaccard@10"], 0.0)


def test_retrieval_partial_overlap():
    # retrieved: q1..q10, gold: {q1, q11} -> 1 overlap
    retrieved = [f"q{i}" for i in range(1, 11)]
    gold = {"q1", "q11"}
    m = compute_retrieval_metrics(retrieved, gold)
    # intersection=1, union=10+1=11, P=1/10, R=1/2, J=1/11
    assert close(m["P@10"], 1 / 10)
    assert close(m["R@10"], 1 / 2)
    assert close(m["Jaccard@10"], 1 / 11)


def test_retrieval_gold_larger_than_10():
    # gold has 15 quotes, we retrieve 10 of them
    retrieved = [f"q{i}" for i in range(10)]
    gold = {f"q{i}" for i in range(15)}
    m = compute_retrieval_metrics(retrieved, gold)
    # intersection=10, union=15, P=10/10=1, R=10/15, J=10/15
    assert close(m["P@10"], 1.0)
    assert close(m["R@10"], 10 / 15)
    assert close(m["Jaccard@10"], 10 / 15)


# ===========================================================================
# aggregate_retrieval
# ===========================================================================

def test_aggregate_all_skipped():
    result = aggregate_retrieval([None, None], [0, 0])
    assert result["n_questions"] == 0
    assert result["n_skipped"] == 2


def test_aggregate_mean_gold_set_size():
    per_q = [
        compute_retrieval_metrics(["q1"] * 10, {"q1"}),
        compute_retrieval_metrics(["q2"] * 10, {"q2", "q3"}),
    ]
    result = aggregate_retrieval(per_q, gold_set_sizes=[1, 2])
    assert close(result["mean_gold_set_size"], 1.5)
    assert result["n_questions"] == 2
    assert result["n_skipped"] == 0


# ===========================================================================
# bootstrap_ci
# ===========================================================================

def test_bootstrap_empty():
    result = bootstrap_ci([])
    assert math.isnan(result["mean"])
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])


def test_bootstrap_constant():
    # All same value -> CI collapses to that value
    result = bootstrap_ci([0.5] * 100, n=500)
    assert close(result["mean"], 0.5)
    assert close(result["ci_low"], 0.5, tol=1e-6)
    assert close(result["ci_high"], 0.5, tol=1e-6)


def test_bootstrap_ci_ordering():
    # ci_low <= mean <= ci_high
    import random
    rng = random.Random(0)
    values = [rng.gauss(0.5, 0.1) for _ in range(200)]
    result = bootstrap_ci(values, n=1000)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bootstrap_reproducible():
    values = [float(i) / 100 for i in range(100)]
    r1 = bootstrap_ci(values, seed=42)
    r2 = bootstrap_ci(values, seed=42)
    assert r1 == r2


if __name__ == "__main__":
    # normalize
    test_normalize_lowercase()
    test_normalize_strips_articles()
    test_normalize_strips_punctuation()
    test_normalize_collapses_whitespace()
    test_normalize_empty()
    # answer metrics
    test_both_empty()
    test_pred_empty_ref_nonempty()
    test_pred_nonempty_ref_empty()
    test_exact_match()
    test_exact_match_after_normalization()
    test_partial_overlap()
    test_no_overlap()
    test_repeated_tokens_multiset()
    test_article_stripping_in_metrics()
    # retrieval metrics
    test_retrieval_empty_gold_returns_none()
    test_retrieval_full_recall()
    test_retrieval_no_overlap()
    test_retrieval_partial_overlap()
    test_retrieval_gold_larger_than_10()
    # aggregate
    test_aggregate_all_skipped()
    test_aggregate_mean_gold_set_size()
    # bootstrap
    test_bootstrap_empty()
    test_bootstrap_constant()
    test_bootstrap_ci_ordering()
    test_bootstrap_reproducible()
    print("\nAll metrics tests passed.")
