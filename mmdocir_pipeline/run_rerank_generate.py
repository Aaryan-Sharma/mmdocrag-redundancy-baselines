"""
Part 2 driver — rerank -> select top-k -> generate answers.
===========================================================
Reads Part 1's retrieval JSONL, rehydrates candidate images from the layouts
parquet (by row_index), reranks with the chosen reranker, keeps top-k, and
generates an answer per query. Writes a response JSONL consumed by eval_metrics.py.

Run:
    python run_rerank_generate.py --reranker ziprerank --gen_mode multimodal --topk 5
    python run_rerank_generate.py --reranker mmr5       --gen_mode hybrid     --topk 10
    python run_rerank_generate.py --reranker identity   --gen_mode pure-text         # baseline
    python run_rerank_generate.py --retrieval_file retrieval/ColQwen_image_top20.jsonl --limit 20

Swap the reranker with --reranker {ziprerank|mmr5|vlm|identity}. Nothing else changes.

Each output line:
    {
      qid, doc_name, domain, question, answer (gold), question_type,
      reranker, gen_mode, topk,
      gold_page_id, layout_mapping,                 # carried for metrics
      reranked: [ {rank, idx, layout_id, page_id, bbox, score, is_gold} ... ],  # full reorder
      selected: [ same fields, top-k ],             # what the generator saw
      prediction: "..."                             # generated answer
    }
"""
import argparse
import io
import json

from PIL import Image
from tqdm import tqdm

import config
from mmdocir_data import load_layouts_parquet
from rerankers import Candidate, get_reranker
from generator import AnswerGenerator


def _reranker_model_id(name):
    return {
        "ziprerank": config.ZIPRERANK_MODEL,
        "zip": config.ZIPRERANK_MODEL,
        "mmr5": config.MMR5_MODEL, "mm-r5": config.MMR5_MODEL, "mm5": config.MMR5_MODEL,
        "vlm": config.VLM_RERANK_MODEL, "rankgpt": config.VLM_RERANK_MODEL,
    }.get(name.lower())


def parse_args():
    p = argparse.ArgumentParser(description="MMDocIR Part 2: rerank + generate")
    p.add_argument("--reranker", default=config.RERANKER,
                   choices=["ziprerank", "mmr5", "mm5", "vlm", "rankgpt", "identity"])
    p.add_argument("--gen_mode", default=config.GENERATION_MODE,
                   choices=["multimodal", "pure-text", "hybrid"])
    p.add_argument("--topk", type=int, default=config.TOPK)
    p.add_argument("--max_images", type=int, default=config.MAX_IMAGES)
    p.add_argument("--retrieval_file", default=None,
                   help="Part-1 output (default: retrieval/ColQwen_image_top20.jsonl)")
    p.add_argument("--generator_model", default=config.GENERATOR_MODEL)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--limit", type=int, default=-1, help="process first N queries (debug)")
    p.add_argument("--no_generate", action="store_true",
                   help="only rerank + select (skip answer generation)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_retrieval(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def image_from_row(layouts_df, row_index):
    b = layouts_df.iloc[row_index]["image_binary"]
    return Image.open(io.BytesIO(b)).convert("RGB")


def main():
    args = parse_args()
    retrieval_file = args.retrieval_file or config.retrieval_output_path("ColQwen", "image", config.RETRIEVAL_TOPN)
    out_path = args.out or config.response_output_path(args.reranker, args.gen_mode, args.topk)

    print(f"Loading retrieval results: {retrieval_file}")
    data = load_retrieval(retrieval_file)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"  {len(data):,} queries")

    print(f"Loading layouts parquet (image rehydration): {config.LAYOUTS_PARQUET}")
    layouts_df = load_layouts_parquet(config.LAYOUTS_PARQUET)

    # --- build reranker (the swap point) ---
    need_images = (args.reranker != "identity") or (args.gen_mode in ("multimodal", "hybrid"))
    if args.reranker == "identity":
        reranker = get_reranker("identity")
    else:
        reranker = get_reranker(args.reranker, model_id=_reranker_model_id(args.reranker),
                                device=args.device, max_images=args.max_images)
    print(f"Reranker: {reranker.name}")

    generator = None
    if not args.no_generate:
        generator = AnswerGenerator(args.generator_model, device=args.device,
                                    max_new_tokens=config.GEN_MAX_NEW_TOKENS)
        print(f"Generator: {args.generator_model} (mode={args.gen_mode})")

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for q in tqdm(data, desc="rerank+gen"):
            raw_cands = q["candidates"]

            # rehydrate images only if some stage needs them
            cands = []
            for i, c in enumerate(raw_cands):
                img = image_from_row(layouts_df, c["row_index"]) if need_images else None
                cands.append(Candidate(idx=i, text=c.get("text", ""), _image=img))

            # --- rerank ---
            order = reranker.rerank(q["question"], cands)

            reranked = []
            for rank, i in enumerate(order):
                c = raw_cands[i]
                reranked.append({
                    "rank": rank, "idx": i,
                    "layout_id": c["layout_id"], "page_id": c["page_id"],
                    "bbox": c["bbox"], "row_index": c["row_index"],
                    "score": c.get("score"), "is_gold": c.get("is_gold", False),
                })
            selected = reranked[: args.topk]

            # --- generate ---
            prediction = ""
            if generator is not None:
                topk_cands = [cands[s["idx"]] for s in selected]
                prediction = generator.generate(q["question"], topk_cands, mode=args.gen_mode)

            fout.write(json.dumps({
                "qid": q["qid"], "doc_name": q["doc_name"], "domain": q["domain"],
                "question": q["question"], "answer": q.get("answer", ""),
                "question_type": q.get("question_type", ""),
                "reranker": reranker.name, "gen_mode": args.gen_mode, "topk": args.topk,
                "gold_page_id": q.get("gold_page_id", []),
                "layout_mapping": q.get("layout_mapping", []),
                "reranked": reranked, "selected": selected,
                "prediction": prediction,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nWrote {n_written:,} responses -> {out_path}")
    print(f"Next: python eval_metrics.py --response_file {out_path}")


if __name__ == "__main__":
    main()
