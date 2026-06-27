"""
Failure analysis for the MMDocIR rerank+generate responses.

Localizes every query's outcome to a pipeline stage and mines failure patterns.

Outcome taxonomy (per query):
  CORRECT          answer matches gold (substring match, the MMDocIR/DDP metric)
  RETRIEVER_MISS   wrong AND gold region not in top-20 at all   (ColQwen failed)
  SELECTION_MISS   wrong AND gold in top-20 but not in top-k    (reranker/selection failed)
  GENERATION_MISS  wrong AND gold IS in top-k                   (generator had evidence, failed)

"gold in top-N" = a candidate's bbox overlaps (area > 0) a gold region on the same page.

Run:
  python failure_analysis.py --response_file response/ziprerank_multimodal_top5_response.jsonl
  python failure_analysis.py --response_file <f> --dump_examples 5

Outputs:
  <stem>_failure_diagnostics.jsonl   per-query diagnostics + category
  <stem>_failure_examples.txt        sampled readable cases per category
  printed report                     overall + by domain + by question_type + paradox + patterns
"""
import argparse
import json
import re
import string
from collections import defaultdict, Counter


# --------------------------------------------------------------------------- #
# answer matching
# --------------------------------------------------------------------------- #
def normalize(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def tokens(s):
    return normalize(s).split()

def answer_metrics(pred, gold):
    p, g = normalize(pred), normalize(gold)
    pt, gt = p.split(), g.split()
    em = float(p == g)
    substr = float(bool(g) and g in p)           # gold contained in prediction (DDP metric)
    if not pt or not gt:
        f1 = float(p == g)
    else:
        common = 0; tmp = list(gt)
        for t in pt:
            if t in tmp: common += 1; tmp.remove(t)
        prec = common / len(pt); rec = common / len(gt)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return em, f1, substr


# --------------------------------------------------------------------------- #
# geometry: does a candidate overlap a gold region?
# --------------------------------------------------------------------------- #
def overlap_area(b1, b2):
    t1, l1, b1b, r1 = b1; t2, l2, b2b, r2 = b2
    it, il = max(t1, t2), max(l1, l2)
    ib, ir = min(b1b, b2b), min(r1, r2)
    return (ib - it) * (ir - il) if (it < ib and il < ir) else 0.0

def cand_hits_gold(cand, layout_mapping):
    for gt in layout_mapping:
        if cand["page_id"] == gt["page"] and overlap_area(cand["bbox"], gt["bbox"]) > 0:
            return True
    return False

def best_gold_rank(reranked, layout_mapping):
    """Lowest rank (0-best) at which a gold-overlapping candidate appears; None if never."""
    for c in reranked:
        if cand_hits_gold(c, layout_mapping):
            return c["rank"]
    return None


# --------------------------------------------------------------------------- #
def diagnose(row, topk):
    pred, gold = row.get("prediction", ""), row.get("answer", "")
    em, f1, substr = answer_metrics(pred, gold)
    correct = bool(substr)

    lm = row.get("layout_mapping", [])
    reranked = sorted(row.get("reranked", []), key=lambda c: c["rank"])
    selected = [c for c in reranked[:topk]]

    gold_rank = best_gold_rank(reranked, lm)              # None => retriever miss
    gold_in_top20 = gold_rank is not None
    gold_in_topk = gold_rank is not None and gold_rank < topk
    gold_pages = set(g["page"] for g in lm)
    sel_pages = set(c["page_id"] for c in selected)
    gold_page_in_topk = bool(gold_pages & sel_pages)

    # category
    if correct:
        cat = "CORRECT"
    elif not gold_in_top20:
        cat = "RETRIEVER_MISS"
    elif not gold_in_topk:
        cat = "SELECTION_MISS"
    else:
        cat = "GENERATION_MISS"

    # extra signals
    pred_n = normalize(pred)
    q_n = normalize(row.get("question", ""))
    pred_copies_q = bool(pred_n) and pred_n in q_n
    sel_scores = [c.get("score") for c in selected if c.get("score") is not None]
    score_std = (max(sel_scores) - min(sel_scores)) if sel_scores else 0.0
    identity = [c["idx"] for c in reranked] == list(range(len(reranked)))

    return {
        "qid": row.get("qid"), "domain": row.get("domain", "?"),
        "question_type": _qtype(row.get("question_type")),
        "category": cat, "correct": correct,
        "em": em, "f1": round(f1, 3), "substr": substr,
        "gold_rank": gold_rank, "gold_in_top20": gold_in_top20,
        "gold_in_topk": gold_in_topk, "gold_page_in_topk": gold_page_in_topk,
        "n_gold_regions": len(lm), "n_gold_pages": len(gold_pages),
        "pred_empty": not pred_n, "pred_copies_question": pred_copies_q,
        "pred_len": len(pred_n.split()), "gold_len": len(normalize(gold).split()),
        "score_spread": round(score_std, 3), "identity_order": identity,
        "question": row.get("question", ""), "gold": gold, "prediction": pred,
    }


def _qtype(qt):
    if isinstance(qt, list): return ",".join(qt)
    if isinstance(qt, str):
        m = re.findall(r"[A-Za-z][A-Za-z/ ]+", qt)
        return ",".join(t.strip() for t in m) if m else qt
    return str(qt)


# --------------------------------------------------------------------------- #
def pct(n, d): return round(100 * n / d, 1) if d else 0.0

def breakdown(diags, key):
    out = {}
    groups = defaultdict(list)
    for d in diags: groups[d[key]].append(d)
    for g, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(items)
        cats = Counter(x["category"] for x in items)
        out[g] = {
            "n": n, "acc": pct(sum(x["correct"] for x in items), n),
            "RETRIEVER_MISS": pct(cats["RETRIEVER_MISS"], n),
            "SELECTION_MISS": pct(cats["SELECTION_MISS"], n),
            "GENERATION_MISS": pct(cats["GENERATION_MISS"], n),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response_file", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dump_examples", type=int, default=4, help="sampled cases per category")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.response_file, encoding="utf-8")]
    diags = [diagnose(r, args.topk) for r in rows]
    N = len(diags)
    cats = Counter(d["category"] for d in diags)

    # ---- overall ----
    print(f"\n=== {args.response_file}  (N={N}, top-{args.topk}) ===\n")
    print("OUTCOME TAXONOMY")
    for c in ["CORRECT", "RETRIEVER_MISS", "SELECTION_MISS", "GENERATION_MISS"]:
        print(f"  {c:16s} {cats[c]:5d}  ({pct(cats[c], N)}%)")
    wrong = N - cats["CORRECT"]
    print(f"  {'-- of failures:':16s}")
    for c in ["RETRIEVER_MISS", "SELECTION_MISS", "GENERATION_MISS"]:
        print(f"     {c:14s} {pct(cats[c], wrong)}% of all errors")

    # ---- retrieval-reasoning paradox ----
    g_in = [d for d in diags if d["gold_in_topk"]]
    g_out = [d for d in diags if not d["gold_in_topk"]]
    print("\nRETRIEVAL-REASONING PARADOX")
    print(f"  gold in top-{args.topk}:     {len(g_in)} ({pct(len(g_in), N)}%) | acc = {pct(sum(d['correct'] for d in g_in), len(g_in))}%")
    print(f"  gold NOT in top-{args.topk}: {len(g_out)} ({pct(len(g_out), N)}%) | acc = {pct(sum(d['correct'] for d in g_out), len(g_out))}%")
    print(f"  gold retrieved into top-20 at all: {pct(sum(d['gold_in_top20'] for d in diags), N)}%")

    # ---- by domain ----
    print("\nBY DOMAIN  (n | acc | %Retr-miss | %Sel-miss | %Gen-miss)")
    for g, v in breakdown(diags, "domain").items():
        print(f"  {g[:34]:34s} {v['n']:4d} | {v['acc']:5.1f} | {v['RETRIEVER_MISS']:5.1f} | {v['SELECTION_MISS']:5.1f} | {v['GENERATION_MISS']:5.1f}")

    # ---- by question type ----
    print("\nBY QUESTION TYPE  (n | acc | %Retr-miss | %Sel-miss | %Gen-miss)")
    for g, v in breakdown(diags, "question_type").items():
        print(f"  {g[:34]:34s} {v['n']:4d} | {v['acc']:5.1f} | {v['RETRIEVER_MISS']:5.1f} | {v['SELECTION_MISS']:5.1f} | {v['GENERATION_MISS']:5.1f}")

    # ---- patterns within failure types ----
    gen = [d for d in diags if d["category"] == "GENERATION_MISS"]
    sel = [d for d in diags if d["category"] == "SELECTION_MISS"]
    print("\nGENERATION_MISS PATTERNS (had the evidence, still wrong)")
    if gen:
        print(f"  count={len(gen)} | pred copies question: {pct(sum(d['pred_copies_question'] for d in gen), len(gen))}%"
              f" | empty pred: {pct(sum(d['pred_empty'] for d in gen), len(gen))}%")
        print(f"  multi-region gold (>1): {pct(sum(d['n_gold_regions'] > 1 for d in gen), len(gen))}%"
              f" | multi-page gold (>1): {pct(sum(d['n_gold_pages'] > 1 for d in gen), len(gen))}%")
    print("SELECTION_MISS PATTERNS (gold retrieved but not in top-k)")
    if sel:
        ranks = [d["gold_rank"] for d in sel]
        within10 = pct(sum(r < 10 for r in ranks), len(sel))
        print(f"  count={len(sel)} | gold rank: min={min(ranks)} median={sorted(ranks)[len(ranks)//2]} max={max(ranks)}"
              f" | would be recovered by top-10: {within10}%")

    # ---- reranking effect ----
    ident = [d for d in diags if d["identity_order"]]
    print("\nRERANKER ACTIVITY")
    print(f"  identity-order (unchanged from ColQwen): {pct(len(ident), N)}%"
          f" | acc when reranked: {pct(sum(d['correct'] for d in diags if not d['identity_order']), N - len(ident))}%"
          f" | acc when identity: {pct(sum(d['correct'] for d in ident), len(ident))}%")
    cor = [d['score_spread'] for d in diags if d['correct']]
    inc = [d['score_spread'] for d in diags if not d['correct']]
    if cor and inc:
        print(f"  mean top-{args.topk} score spread:  correct={round(sum(cor)/len(cor),2)}  incorrect={round(sum(inc)/len(inc),2)}")

    # ---- write diagnostics ----
    stem = args.response_file.rsplit(".", 1)[0]
    with open(stem + "_failure_diagnostics.jsonl", "w", encoding="utf-8") as f:
        for d in diags:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- sampled examples per category ----
    with open(stem + "_failure_examples.txt", "w", encoding="utf-8") as f:
        for c in ["RETRIEVER_MISS", "SELECTION_MISS", "GENERATION_MISS"]:
            ex = [d for d in diags if d["category"] == c][: args.dump_examples]
            f.write(f"\n{'='*70}\n{c}  (showing {len(ex)})\n{'='*70}\n")
            for d in ex:
                f.write(f"[qid {d['qid']}] {d['domain']} | type={d['question_type']} | gold_rank={d['gold_rank']}\n")
                f.write(f"  Q: {d['question']}\n  GOLD: {d['gold']}\n  PRED: {d['prediction']}\n\n")

    print(f"\nWrote: {stem}_failure_diagnostics.jsonl  and  {stem}_failure_examples.txt")


if __name__ == "__main__":
    main()
