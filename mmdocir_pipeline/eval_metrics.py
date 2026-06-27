"""
Evaluation (Part 2)
==================
Computes two metric groups over the response JSONL from run_rerank_generate.py:

  ANSWER (generation quality), token-level, prediction vs gold answer:
      EM, F1, Precision, Recall, Jaccard
  RETRIEVAL (rerank + selection quality), the official MMDocIR layout metric:
      layout recall@k = normalised bbox-overlap area of the SELECTED top-k
      reranked quotes vs the gold regions (layout_mapping)

Both overall and per-domain. Pure set/string ops + the shared mmdocir_metrics.

Run:
    python eval_metrics.py --response_file response/ziprerank_multimodal_top5_response.jsonl
    python eval_metrics.py --response_file <file> --recall_ks 1 3 5 10
"""
import argparse
import json
import re
import string
from collections import defaultdict

from mmdocir_metrics import top_k_indices, recall_layout


# --------------------------------------------------------------------------- #
# Answer metrics (token-level)
# --------------------------------------------------------------------------- #
def normalize(s: str):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def answer_scores(pred: str, gold: str):
    p, g = normalize(pred), normalize(gold)
    ps, gs = set(p), set(g)
    em = float(p == g)
    if not p and not g:
        return dict(em=1.0, f1=1.0, precision=1.0, recall=1.0, jaccard=1.0)
    if not p or not g:
        return dict(em=em, f1=0.0, precision=0.0, recall=0.0, jaccard=0.0)
    # token-multiset overlap for P/R/F1 (SQuAD-style)
    common = 0
    gtmp = list(g)
    for t in p:
        if t in gtmp:
            common += 1; gtmp.remove(t)
    precision = common / len(p)
    recall = common / len(g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    jaccard = len(ps & gs) / len(ps | gs)
    return dict(em=em, f1=f1, precision=precision, recall=recall, jaccard=jaccard)


# --------------------------------------------------------------------------- #
# Retrieval metric (post-rerank layout recall@k) — official area-overlap
# --------------------------------------------------------------------------- #
def layout_recall_at_k(selected_or_reranked, layout_mapping, k):
    """
    selected_or_reranked : list of reranked candidate dicts (have page_id, bbox, score-order)
    We rank by their existing order (already best-first), take top-k, area-overlap recall.
    """
    items = selected_or_reranked[:k]
    # recall_layout expects local positions + a parallel (row, page, x1,y1,x2,y2) index list
    layout_index_list = [(it["row_index"], it["page_id"], *it["bbox"]) for it in items]
    topk_local = list(range(len(items)))
    return recall_layout(topk_local, layout_index_list, layout_mapping)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="MMDocIR Part 2 evaluation")
    p.add_argument("--response_file", required=True)
    p.add_argument("--recall_ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.response_file, encoding="utf-8")]
    print(f"Loaded {len(rows):,} responses from {args.response_file}")

    ans_keys = ["em", "f1", "precision", "recall", "jaccard"]
    overall_ans = {k: 0.0 for k in ans_keys}
    overall_rec = {k: 0.0 for k in args.recall_ks}
    n_ans = 0
    n_rec = 0
    by_domain = defaultdict(lambda: {"n_ans": 0, "n_rec": 0,
                                     "ans": {k: 0.0 for k in ans_keys},
                                     "rec": {k: 0.0 for k in args.recall_ks}})

    has_predictions = any(r.get("prediction") for r in rows)

    for r in rows:
        dom = r.get("domain", "?")
        # answer metrics (only if generation was run)
        if has_predictions:
            sc = answer_scores(r.get("prediction", ""), r.get("answer", ""))
            for k in ans_keys:
                overall_ans[k] += sc[k]; by_domain[dom]["ans"][k] += sc[k]
            n_ans += 1; by_domain[dom]["n_ans"] += 1

        # retrieval metrics (need gold regions)
        lm = r.get("layout_mapping", [])
        reranked = r.get("reranked", [])
        if lm and reranked:
            for k in args.recall_ks:
                rec = layout_recall_at_k(reranked, lm, k)
                overall_rec[k] += rec; by_domain[dom]["rec"][k] += rec
            n_rec += 1; by_domain[dom]["n_rec"] += 1

    # ---- assemble ----
    report = {"response_file": args.response_file, "n": len(rows)}
    if has_predictions and n_ans:
        report["answer"] = {k: round(100 * overall_ans[k] / n_ans, 2) for k in ans_keys}
    if n_rec:
        report["layout_recall"] = {f"@{k}": round(100 * overall_rec[k] / n_rec, 2) for k in args.recall_ks}
    report["by_domain"] = {}
    for dom, d in by_domain.items():
        entry = {}
        if has_predictions and d["n_ans"]:
            entry["answer"] = {k: round(100 * d["ans"][k] / d["n_ans"], 2) for k in ans_keys}
        if d["n_rec"]:
            entry["layout_recall"] = {f"@{k}": round(100 * d["rec"][k] / d["n_rec"], 2) for k in args.recall_ks}
        report["by_domain"][dom] = entry

    # ---- print ----
    print("\n================ OVERALL ================")
    if "answer" in report:
        print("Answer:", "  ".join(f"{k.upper()}={v}" for k, v in report["answer"].items()))
    if "layout_recall" in report:
        print("Layout recall (area-overlap):",
              "  ".join(f"{k}={v}" for k, v in report["layout_recall"].items()))
    print("\n---------------- by domain ----------------")
    for dom, e in report["by_domain"].items():
        bits = []
        if "answer" in e:        bits.append(f"F1={e['answer']['f1']}")
        if "layout_recall" in e: bits.append("R@" + " ".join(f"{kk[1:]}={vv}" for kk, vv in e['layout_recall'].items()))
        print(f"  {dom:35s} {'  '.join(bits)}")

    out = args.out or (args.response_file.rsplit(".", 1)[0] + "_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
