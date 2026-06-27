"""
Exact MMDocIR evaluation primitives, vendored verbatim (logic-identical) from the
official repo: https://github.com/MMDocRAG/MMDocIR/blob/main/metric_eval.py

Shared by Part 1 (retrieval sanity recall) and Part 2 (final retrieval metrics) so
that numbers are directly comparable to the published MMDocIR layout-retrieval results.

KEY SUBTLETY — layout relevance is AREA-OVERLAP, not exact id matching:
  For each retrieved layout we sum the bbox-intersection area against every gold
  region in `layout_mapping` that lies on the same page, then normalise by total
  gold area. This is `recall_layout` below. Do not replace it with set overlap.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Ranking helpers
# --------------------------------------------------------------------------- #
def top_k_indices(scores, k):
    """Return indices of the top-k scores (descending). Falls back to all if k>len."""
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if k <= len(scores):
        return [index for index, _ in indexed_scores[:k]]
    return [index for index, _ in indexed_scores]


# --------------------------------------------------------------------------- #
# Set-based retrieval metrics (used for PAGE-level; handy for layout-id analysis)
# --------------------------------------------------------------------------- #
def precision(retrieved, ground_truth):
    tp = len(set(retrieved) & set(ground_truth))
    return tp / len(retrieved) if len(retrieved) > 0 else 0


def recall(retrieved, ground_truth):
    tp = len(set(retrieved) & set(ground_truth))
    return tp / len(ground_truth) if ground_truth else 0


def ndcg(retrieved, ground_truth):
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(retrieved), len(ground_truth))))
    dcg = sum(1 / np.log2(i + 2) for i in range(len(retrieved)) if retrieved[i] in ground_truth)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0


def average_precision(retrieved, ground_truth):
    tp, ap = 0, 0.0
    for i, item in enumerate(retrieved):
        if item in ground_truth:
            tp += 1
            ap += tp / (i + 1)
    return ap / tp if tp else 0


def mean_reciprocal_rank(retrieved, ground_truth):
    for i, item in enumerate(retrieved):
        if item in ground_truth:
            return 1 / (i + 1)
    return 0


# --------------------------------------------------------------------------- #
# Layout-level metric: normalised bbox-overlap recall (the official one)
# --------------------------------------------------------------------------- #
def calculate_overlap_score(bbox1, bbox2):
    """Intersection area of two bboxes. Both unpacked with the same coordinate
    ordering as the official code (the absolute ordering is irrelevant as long as
    it is consistent on both sides)."""
    top1, left1, bottom1, right1 = bbox1
    top2, left2, bottom2, right2 = bbox2
    inter_top    = max(top1, top2)
    inter_left   = max(left1, left2)
    inter_bottom = min(bottom1, bottom2)
    inter_right  = min(right1, right2)
    if inter_top < inter_bottom and inter_left < inter_right:
        return (inter_bottom - inter_top) * (inter_right - inter_left)
    return 0.0


def recall_layout(layout_topk, layout_indices, layout_mapping):
    """
    layout_topk    : list of LOCAL positions (into this query's candidate list)
    layout_indices : parallel list; each item is (row_index, page_id, x1, y1, x2, y2)
    layout_mapping : gold regions [{"page": int, "page_size": [...], "bbox": [x1,y1,x2,y2]}, ...]
    Returns normalised overlap recall in [0, 1].
    """
    recall_area = 0.0
    for layout_id in layout_topk:
        _, page_id, x1, y1, x2, y2 = layout_indices[layout_id]
        for gt_layout in layout_mapping:
            if page_id == gt_layout["page"]:
                recall_area += calculate_overlap_score([x1, y1, x2, y2], gt_layout["bbox"])

    gt_area = 0.0
    for gt_layout in layout_mapping:
        top, left, bottom, right = gt_layout["bbox"]
        gt_area += (bottom - top) * (right - left)

    if gt_area == 0:
        return 0.0
    return recall_area / gt_area


# --------------------------------------------------------------------------- #
# Convenience: single-bbox overlap (for per-candidate gold labelling in Part 1)
# --------------------------------------------------------------------------- #
def candidate_gold_overlap(page_id, bbox, layout_mapping):
    """Total intersection area of one candidate layout against all gold regions
    on its page. >0 means the candidate touches a gold region."""
    area = 0.0
    for gt_layout in layout_mapping:
        if page_id == gt_layout["page"]:
            area += calculate_overlap_score(bbox, gt_layout["bbox"])
    return area
