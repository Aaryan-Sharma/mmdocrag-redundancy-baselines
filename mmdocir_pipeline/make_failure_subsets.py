"""
Build the three failure-browser subsets from a rerank response file.

Categories (gold membership recomputed from bbox overlap with layout_mapping,
so the notebook's gold/non-gold colouring is accurate even if stored is_gold is stale):

  gen_fail        wrong answer AND gold region is in the selected top-k   (generator had it)
  near_miss       wrong answer AND gold in top-20 but NOT in top-k        (selection just missed)
  hurt_by_rerank  retriever's #1 (idx=0) was gold, but reranker's #1 (rank=0) is NOT gold
                  (the reranker demoted a gold that ColQwen had at the top)

Each output record is the FULL response record (with is_gold fixed on every
reranked/selected candidate) so fail.ipynb can browse it directly.

Run:
  python make_failure_subsets.py --response_file response/ziprerank_multimodal_top5_response.jsonl --prefix ziprerank
  # -> ziprerank_gen_fail.jsonl  ziprerank_near_miss.jsonl  ziprerank_hurt_by_rerank.jsonl
"""
import argparse, json, re, string


def normalize(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def correct(pred, gold):
    g = normalize(gold)
    return bool(g) and g in normalize(pred)          # substring match (MMDocIR/DDP metric)

def overlap(b1, b2):
    t1,l1,bo1,r1=b1; t2,l2,bo2,r2=b2
    it,il=max(t1,t2),max(l1,l2); ib,ir=min(bo1,bo2),min(r1,r2)
    return (ib-it)*(ir-il) if (it<ib and il<ir) else 0.0

def hits_gold(c, lm):
    return any(c["page_id"]==g["page"] and overlap(c["bbox"], g["bbox"])>0 for g in lm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response_file", required=True)
    ap.add_argument("--prefix", default=None, help="output filename prefix (default: from reranker field)")
    ap.add_argument("--topk", type=int, default=None, help="override; else uses record's topk / len(selected)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.response_file, encoding="utf-8")]
    prefix = args.prefix or (rows[0].get("reranker", "model").lower().replace("-", "") if rows else "model")

    gen_fail, near_miss, hurt = [], [], []
    for r in rows:
        lm = r.get("layout_mapping", [])
        reranked = sorted(r.get("reranked", []), key=lambda c: c["rank"])
        topk = args.topk or r.get("topk") or len(r.get("selected", [])) or 5

        # fix is_gold on every candidate (in-place) for accurate notebook colouring
        for c in reranked:
            c["is_gold"] = hits_gold(c, lm)
        sel = reranked[:topk]
        r["selected"] = sel  # keep selected consistent with fixed flags + order

        ok = correct(r.get("prediction", ""), r.get("answer", ""))
        gold_in_topk  = any(c["is_gold"] for c in sel)
        gold_in_top20 = any(c["is_gold"] for c in reranked)

        if not ok and gold_in_topk:
            gen_fail.append(r)
        elif not ok and gold_in_top20:          # in pool, missed selection
            near_miss.append(r)

        idx0 = next((c for c in reranked if c["idx"] == 0), None)
        rank0 = next((c for c in reranked if c["rank"] == 0), None)
        if idx0 and rank0 and idx0["is_gold"] and not rank0["is_gold"]:
            hurt.append(r)

    def dump(name, data):
        path = f"{prefix}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path:42s} {len(data):5d} records")

    print(f"From {args.response_file}  (N={len(rows)}, top-{rows[0].get('topk', '?') if rows else '?'}):")
    dump("gen_fail", gen_fail)
    dump("near_miss", near_miss)
    dump("hurt_by_rerank", hurt)


if __name__ == "__main__":
    main()
