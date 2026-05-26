"""
Track 1 — Answer quality metrics (SQuAD-style token overlap).

All metrics operate on a single (prediction, reference) string pair
where the reference is already the output of data_loader.parse_answer().
"""
import re
import string
from collections import Counter

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """
    Lowercase, strip punctuation, remove articles, collapse whitespace.
    Matches SQuAD evaluation normalization.
    """
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _token_counts(s: str) -> Counter:
    return Counter(s.split())


def compute_answer_metrics(prediction: str, reference: str) -> dict:
    """
    Compute EM, F1, Precision, Recall, Jaccard on normalized token bags.

    Edge cases
    ----------
    Both empty after normalization  -> all metrics = 1.
    Only prediction empty           -> P/R/F1/Jaccard = 0, EM = 0.
    Only reference empty            -> P/R/F1/Jaccard = 0, EM = 0.
    Identical after normalization   -> all metrics = 1 (covered by EM path).
    """
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    # Both empty
    if not pred_norm and not ref_norm:
        return {"EM": 1, "F1": 1.0, "Precision": 1.0, "Recall": 1.0, "Jaccard": 1.0}

    em = int(pred_norm == ref_norm)

    # One side empty
    if not pred_norm or not ref_norm:
        return {"EM": em, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, "Jaccard": 0.0}

    pred_counts = _token_counts(pred_norm)
    ref_counts = _token_counts(ref_norm)

    # Multiset intersection: sum of min(count_pred, count_ref) per token
    intersection = sum((pred_counts & ref_counts).values())
    # Multiset union: sum of max(count_pred, count_ref) per token
    union = sum((pred_counts | ref_counts).values())

    n_pred = sum(pred_counts.values())
    n_ref = sum(ref_counts.values())

    precision = intersection / n_pred  # n_pred > 0 guaranteed above
    recall = intersection / n_ref      # n_ref  > 0 guaranteed above
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = intersection / union if union > 0 else 0.0

    return {
        "EM": em,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Jaccard": jaccard,
    }
