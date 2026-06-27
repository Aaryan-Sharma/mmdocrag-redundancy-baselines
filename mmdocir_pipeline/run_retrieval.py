"""
Part 1 driver — ColQwen retrieval over MMDocIR layout ("quote") candidates.
===========================================================================
Pipeline (this file):
    annotations + layouts.parquet
      -> for each document: build quotes (layouts), encode once with ColQwen
      -> for each query of that document: score vs the doc's quotes (MaxSim)
      -> keep top-N candidates
      -> write one JSONL line per query (metadata + text + row_index; no image bytes)

Output -> retrieval/ColQwen_{mode}_top{N}.jsonl  (consumed by Part 2)

Run:
    python run_retrieval.py
    python run_retrieval.py --mode image --topn 20 --device cuda
    python run_retrieval.py --limit_docs 5 --eval_recall      # quick smoke test + recall

Each output line:
    {
      qid, doc_name, domain, question, answer, question_type,
      gold_page_id, layout_mapping,         # ground truth (for Part-2 metrics)
      retriever, mode, num_doc_layouts,
      candidates: [ {rank, score, local_idx, row_index, layout_id, page_id,
                     type, text, bbox, page_size, image_path,
                     is_gold, gold_overlap}, ... up to N ]
    }
"""
import argparse
import json
import os

from tqdm import tqdm

import config
from mmdocir_data import (
    load_layouts_parquet, load_queries, build_quotes, label_gold,
    group_queries_by_document,
)
from mmdocir_metrics import top_k_indices, recall_layout
from retrievers import get_retriever


def parse_args():
    p = argparse.ArgumentParser(description="MMDocIR Part 1: ColQwen layout retrieval")
    p.add_argument("--retriever", default="ColQwen")
    p.add_argument("--model", default=config.COLQWEN_MODEL)
    p.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["image", "vlm_text"],
                   help="how a quote is represented to the retriever")
    p.add_argument("--topn", type=int, default=config.RETRIEVAL_TOPN,
                   help="candidates kept per query (rerank window for Part 2)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--image_bs", type=int, default=config.IMAGE_BS)
    p.add_argument("--query_bs", type=int, default=config.QUERY_BS)
    p.add_argument("--limit_docs", type=int, default=-1, help="process only first N docs (debug)")
    p.add_argument("--eval_recall", action="store_true",
                   help="also report layout recall@{1,5,10} on retrieved candidates")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or config.retrieval_output_path(args.retriever, args.mode, args.topn)

    print(f"Loading layouts parquet: {config.LAYOUTS_PARQUET}")
    layouts_df = load_layouts_parquet(config.LAYOUTS_PARQUET)
    print(f"  {len(layouts_df):,} layouts loaded")

    print(f"Loading queries: {config.ANNOTATIONS}")
    queries = load_queries(config.ANNOTATIONS)
    grouped = group_queries_by_document(queries)
    print(f"  {len(queries):,} queries across {len(grouped):,} documents")

    retriever = get_retriever(
        args.retriever, model_name=args.model, device=args.device,
        image_bs=args.image_bs, query_bs=args.query_bs,
        use_flash_attn=config.USE_FLASH_ATTN,
    )

    # recall accumulators
    recall_at = {1: 0.0, 5: 0.0, 10: 0.0}
    n_eval = 0

    doc_items = list(grouped.items())
    if args.limit_docs > 0:
        doc_items = doc_items[:args.limit_docs]

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for doc_name, g in tqdm(doc_items, desc="documents"):
            doc_queries = g["queries"]
            quotes = build_quotes(layouts_df, g["layout_indices"], load_images=(args.mode == "image"))
            if len(quotes) == 0:
                continue

            # ---- encode this document's candidate quotes once ----
            if args.mode == "image":
                quote_embs = retriever.embed_image_quotes([q.image() for q in quotes])
            else:  # vlm_text
                quote_embs = retriever.embed_text_quotes([q.text for q in quotes])

            # ---- encode the document's queries ----
            query_embs = retriever.embed_queries([q.question for q in doc_queries])

            # ---- score [n_queries, n_quotes] ----
            scores = retriever.score(query_embs, quote_embs)            # tensor
            scores = scores.tolist() if hasattr(scores, "tolist") else scores

            # parallel index list for the official recall metric
            layout_index_list = [q.index_tuple() for q in quotes]

            for qi, query in enumerate(doc_queries):
                q_scores = scores[qi]
                topn_local = top_k_indices(q_scores, args.topn)

                gold_overlaps = label_gold([quotes[li] for li in topn_local], query.layout_mapping)

                candidates = []
                for rank, (li, gov) in enumerate(zip(topn_local, gold_overlaps)):
                    rec = quotes[li].to_record()
                    rec["rank"] = rank
                    rec["score"] = float(q_scores[li])
                    rec["gold_overlap"] = float(gov)
                    rec["is_gold"] = bool(gov > 0)
                    candidates.append(rec)

                fout.write(json.dumps({
                    "qid": query.qid,
                    "doc_name": query.doc_name,
                    "domain": query.domain,
                    "question": query.question,
                    "answer": query.answer,
                    "question_type": query.question_type,
                    "gold_page_id": query.gold_page_id,
                    "layout_mapping": query.layout_mapping,
                    "retriever": retriever.name,
                    "mode": args.mode,
                    "num_doc_layouts": len(quotes),
                    "candidates": candidates,
                }, ensure_ascii=False) + "\n")
                n_written += 1

                # optional recall (uses FULL doc scores -> matches official eval)
                if args.eval_recall and query.layout_mapping:
                    n_eval += 1
                    for k in recall_at:
                        topk_local = top_k_indices(q_scores, k)
                        recall_at[k] += recall_layout(topk_local, layout_index_list, query.layout_mapping)

    print(f"\nWrote {n_written:,} query results -> {out_path}")
    if args.eval_recall and n_eval:
        print("\nLayout recall (normalised overlap), official metric:")
        for k in (1, 5, 10):
            print(f"  recall@{k}: {100 * recall_at[k] / n_eval:.1f}   (over {n_eval} queries w/ gold)")


if __name__ == "__main__":
    main()
