"""
ColPali load + retrieval sanity check.

Gate: run this before any smoke test to verify:
  1. ColPali loads with ZERO unexpected/missing key warnings.
  2. Scores are non-trivial (not all-equal), and the most visually
     relevant candidate ranks #1.

Usage:
    python scripts/colpali_sanity.py --images-dir /tmp/mmdocrag_data --data-dir /tmp

Picks 5 image candidates from dev_20.jsonl question 0, queries with
its question text, and prints the ranked scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image


_COLPALI_KEY_MAPPING = {
    r"^vlm\.language_model\.model\.": "model.model.language_model.",
    r"^vlm\.vision_tower\.": "model.model.vision_tower.",
    r"^vlm\.multi_modal_projector\.": "model.model.multi_modal_projector.",
    r"^embedding_proj_layer\.": "custom_text_proj.",
}


def load_colpali(device: str) -> tuple[ColPali, ColPaliProcessor]:
    print(f"Loading ColPali v1.3-hf on {device} ...")

    # Capture warnings so we can inspect them after load
    caught: list[str] = []
    _orig = warnings.showwarning

    def _capture(msg, category, filename, lineno, file=None, line=None):  # type: ignore[override]
        caught.append(str(msg))
        _orig(msg, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _capture  # type: ignore[assignment]
    try:
        model = ColPali.from_pretrained(
            "vidore/colpali-v1.3-hf",
            torch_dtype=torch.bfloat16,
            device_map=device,
            key_mapping=_COLPALI_KEY_MAPPING,
        ).eval()
    finally:
        warnings.showwarning = _orig  # type: ignore[assignment]

    key_warnings = [
        w for w in caught
        if "unexpected" in w.lower() or "missing" in w.lower()
    ]
    if key_warnings:
        print(f"\nFAIL — {len(key_warnings)} key warning(s) during load:")
        for w in key_warnings[:10]:
            print(f"  {w[:250]}")
        sys.exit(1)

    print("PASS — no unexpected/missing key warnings.\n")
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="Root dir; img_path values resolve relative to this.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing dev_20.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor = load_colpali(args.device)

    # Load question 0 from dev split
    data_path = args.data_dir / "dev_20.jsonl"
    with open(data_path, encoding="utf-8") as fh:
        item = json.loads(fh.readline())

    question = item["question"]
    img_quotes = item.get("img_quotes", [])[:5]
    gold_ids = set(item.get("gold_quotes", []))

    if not img_quotes:
        sys.exit("No img_quotes in first dev item — check field names.")

    print(f"Question: {question[:100]!r}")
    print(f"Gold: {sorted(gold_ids)}\n")

    images = [
        Image.open(args.images_dir / q["img_path"]).convert("RGB")
        for q in img_quotes
    ]

    batch_images = processor.process_images(images).to(model.device)
    batch_query = processor.process_queries([question]).to(model.device)

    with torch.no_grad():
        img_embs = model(**batch_images)
        q_embs = model(**batch_query)

    scores = processor.score_multi_vector(q_embs, img_embs)
    scores_list = scores[0].float().cpu().tolist()

    ranked = sorted(
        zip(scores_list, [q["quote_id"] for q in img_quotes]),
        reverse=True,
    )

    print(f"{'#':<4} {'quote_id':<12} {'score':>8}  gold?")
    print("-" * 34)
    for rank, (score, qid) in enumerate(ranked, 1):
        marker = " <-- GOLD" if qid in gold_ids else ""
        print(f"{rank:<4} {qid:<12} {score:>8.4f}{marker}")

    lo, hi = min(scores_list), max(scores_list)
    spread = hi - lo
    print(f"\nScore range: {lo:.4f} – {hi:.4f}  (spread={spread:.4f})")

    if spread < 1e-4:
        print("FAIL — all scores are (near-)equal; ColPali is not discriminating.")
        sys.exit(1)

    print("PASS — ColPali scores are discriminating.")


if __name__ == "__main__":
    main()
