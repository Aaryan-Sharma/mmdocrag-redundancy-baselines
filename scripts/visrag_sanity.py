"""
VisRAG-Ret sanity check — gate before Pipeline B smoke test.

Verifies:
  1. VisRAG-Ret loads without error.
  2. Scores over 5 candidate images + 1 query are non-flat (spread > 0.01).

Usage:
    python scripts/visrag_sanity.py --images-dir /tmp/mmdocrag_data --data-dir /tmp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from pipeline_b.retriever import _encode, load_models, unload_models, QUERY_PREFIX

    print("Loading VisRAG-Ret ...")
    load_models(device=args.device)
    print("Load complete.\n")

    data_path = args.data_dir / "dev_20.jsonl"
    with open(data_path, encoding="utf-8") as fh:
        item = json.loads(fh.readline())

    question = item["question"]
    img_quotes = item.get("img_quotes", [])[:5]
    gold_ids = set(item.get("gold_quotes", []))

    if not img_quotes:
        sys.exit("No img_quotes in first dev item.")

    print(f"Question: {question[:100]!r}")
    print(f"Gold: {sorted(gold_ids)}\n")

    images = [
        Image.open(args.images_dir / q["img_path"]).convert("RGB")
        for q in img_quotes
    ]

    q_emb = _encode([QUERY_PREFIX + question])   # (1, d)
    img_embs = _encode(images)                    # (n, d)
    scores = (q_emb @ img_embs.T)[0].tolist()

    ranked = sorted(zip(scores, [q["quote_id"] for q in img_quotes]), reverse=True)

    print(f"{'#':<4} {'quote_id':<12} {'score':>8}  gold?")
    print("-" * 34)
    for rank, (score, qid) in enumerate(ranked, 1):
        marker = " <-- GOLD" if qid in gold_ids else ""
        print(f"{rank:<4} {qid:<12} {score:>8.4f}{marker}")

    lo, hi = min(scores), max(scores)
    spread = hi - lo
    print(f"\nScore range: {lo:.4f} – {hi:.4f}  (spread={spread:.4f})")

    unload_models()

    if spread < 0.01:
        print("FAIL — scores are near-uniform; VisRAG-Ret is not discriminating.")
        sys.exit(1)

    print("PASS — VisRAG-Ret scores are discriminating.")


if __name__ == "__main__":
    main()
