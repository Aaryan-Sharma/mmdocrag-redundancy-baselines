"""
Pipeline C retriever: MM-R5 (image reranker) + BGE (text scorer).

Image candidates  -> i2vec/MM-R5 (list-wise VLM reranker, Qwen2.5-VL fine-tuned via GRPO)
Text candidates   -> BAAI/bge-large-en-v1.5 (cosine, sentence-transformers)

Score fusion (per question):
  1. MM-R5 produces a ranked order over the full image pool.
     Pseudo-score: s_img = 1.0 - rank / N_img   (range [0, 1))
     When N_img == 1, skip MM-R5 and assign 1.0 (no ranking possible).
  2. BGE cosine scores over the text pool are min-max normalised to [0, 1]
     with ε=1e-8 guard.  When N_txt == 1 the single candidate gets 1.0.
  3. Both pools are merged and sorted descending; top-k ids returned.

Cache semantics (variant-agnostic for images, variant-agnostic for text):
  Both MM-R5 pseudo-scores and BGE raw cosines are stored in the shared
  score cache keyed by quote_id.  build_variant() never adds image-type
  candidates, so the image pool is identical across all 4 variants for a
  given question: MM-R5 is called at most once per question total.  Text
  twin quote_ids are unique per variant, so each twin is scored once.

Overrides vs. the upstream examples/reranker.py:
  attn_implementation="sdpa"    (no flash-attn on this server)
  do_sample=False, num_beams=1  (greedy decoding per spec)
  max_new_tokens=4096           (chain-of-thought budget)

Score ceiling design note:
  MM-R5 pseudo-scores are capped at 1.0 - 1/N_img (e.g., 0.95 for
  N=20).  A text candidate whose BGE cosine normalises to 1.0 will
  always rank above every image candidate.  In the original variant
  (no text twins) this is moot.  In negative_redundant and
  mixed_redundant, the large number of text twins may lower the BGE
  normalisation range and reduce this effect, but a single high-BGE
  twin can still displace top-ranked images.  If post-eval analysis
  shows pathological text-over-image dominance in redundant variants,
  consider capping BGE normalised scores at a value below 1.0 or
  using a weighted fusion (alpha * img_score + (1-alpha) * txt_score).

Requires Pipeline A venv (envs/requirements_a.txt).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "qwen-vl-utils is required for Pipeline C retriever. "
        "Install it: pip install qwen-vl-utils"
    ) from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MM_R5_MODEL_ID = "i2vec/MM-R5"

_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

_QUESTION_TEMPLATE = (
    "Please rank the following images according to their relevance to the question. "
    "Provide your response in the format: <think>your reasoning process here</think><answer>[image_id_1, image_id_2, ...]</answer> "
    "where the numbers in the list represent the ranking order of images'id from most to least relevant. "
    "Before outputting the answer, you need to analyze each image and provide your analysis process."
    "For example: <think>Image 1 shows the most relevant content because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
    "\nThe question is: {Question}"
    "\n\nThere are {image_num} images, id from 1 to {image_num}, Image ID to image mapping:\n"
)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_mm_r5_model: Qwen2_5_VLForConditionalGeneration | None = None
_mm_r5_processor: Qwen2_5_VLProcessor | None = None
_bge_model: SentenceTransformer | None = None

# Parse statistics — reset by load_models(), readable after unload_models().
_stats: dict[str, int] = {"mm_r5_calls": 0, "mm_r5_fallbacks": 0}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def load_models(device: str = "cuda") -> None:
    """Load MM-R5 and BGE models into module-level singletons."""
    global _mm_r5_model, _mm_r5_processor, _bge_model
    _stats["mm_r5_calls"] = 0
    _stats["mm_r5_fallbacks"] = 0

    print("Loading MM-R5 (i2vec/MM-R5) ...")
    _mm_r5_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MM_R5_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # override: no flash-attn on this server
        device_map="auto",
    ).eval()
    _mm_r5_processor = Qwen2_5_VLProcessor.from_pretrained(MM_R5_MODEL_ID)
    dmap = getattr(_mm_r5_model, "hf_device_map", None) or "auto (single-GPU)"
    print(f"  MM-R5 device map: {dmap}")

    print("Loading BGE-large-en-v1.5 ...")
    _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Pipeline C retriever models loaded.")


def unload_models() -> None:
    """Release GPU memory; print parse statistics before clearing."""
    global _mm_r5_model, _mm_r5_processor, _bge_model
    calls = _stats["mm_r5_calls"]
    fallbacks = _stats["mm_r5_fallbacks"]
    rate = 100.0 * fallbacks / max(1, calls)
    print(f"  MM-R5 stats: {calls} calls, {fallbacks} fallbacks ({rate:.1f}% fallback rate)")
    _mm_r5_model = None
    _mm_r5_processor = None
    _bge_model = None
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax(scores: list[float], eps: float = 1e-8) -> list[float]:
    """Min-max normalise to [0, 1] with ε guard for zero-range pools."""
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < eps:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _score_texts_bge(
    query: str,
    txt_candidates: list[dict],
    q_bge: np.ndarray | None = None,
) -> tuple[list[float], np.ndarray]:
    """Return (raw_cosine_scores, q_bge).  Encodes query if q_bge is None."""
    assert _bge_model is not None
    if q_bge is None:
        q_bge = _bge_model.encode([query], normalize_embeddings=True)
    texts = [c["text"] for c in txt_candidates]
    t_embs = _bge_model.encode(texts, normalize_embeddings=True)
    scores = (q_bge @ t_embs.T)[0]
    return scores.tolist(), q_bge


def _mm_r5_rerank(query: str, img_paths: list[str]) -> list[int]:
    """
    Run MM-R5 on a list of absolute image file paths.

    Returns a 0-indexed list [best, second-best, ..., worst].
    On any parse failure (no <answer> tag, malformed content, out-of-range
    ids), falls back to identity order [0, 1, ..., N-1] and increments
    the fallback counter.
    """
    assert _mm_r5_model is not None and _mm_r5_processor is not None
    N = len(img_paths)
    _stats["mm_r5_calls"] += 1

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _QUESTION_TEMPLATE.format(Question=query, image_num=N),
                },
                # Interleave "Image i: " label + image for each candidate
                *[
                    entry
                    for i, path in enumerate(img_paths)
                    for entry in (
                        {"type": "text", "text": f"\nImage {i + 1}: "},
                        {"type": "image", "image": path},
                    )
                ],
            ],
        },
    ]

    text = _mm_r5_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _mm_r5_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to the device of the first model parameter (safe for device_map="auto")
    model_device = next(_mm_r5_model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        output_ids = _mm_r5_model.generate(
            **inputs,
            do_sample=False,      # greedy per spec
            num_beams=1,
            max_new_tokens=4096,
            use_cache=True,
        )

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    output_text = _mm_r5_processor.decode(new_ids, skip_special_tokens=True)

    match = re.search(r"<answer>\[(.*?)\]</answer>", output_text)
    if match:
        try:
            order_0: list[int] = []
            for tok in match.group(1).strip().split(","):
                tok = tok.strip()
                if not tok:
                    continue
                idx_0 = int(tok) - 1   # 1-indexed → 0-indexed
                if 0 <= idx_0 < N and idx_0 not in order_0:
                    order_0.append(idx_0)
            # Append any indices the model omitted (preserve relative order)
            missing = [i for i in range(N) if i not in order_0]
            order_0.extend(missing)
            return order_0
        except Exception:
            pass  # fall through to fallback

    _stats["mm_r5_fallbacks"] += 1
    print(f"  [MM-R5] parse fallback. Tail of output: {output_text[-300:]!r}")
    return list(range(N))


# ---------------------------------------------------------------------------
# Public interface  (matches the contract used by run_pipeline.py)
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    images_dir: str | Path,
    topk: int = 10,
) -> list[str]:
    """Re-rank candidates for query and return top-k quote_ids."""
    retrieved_ids, _, _ = rerank_with_cache(query, candidates, images_dir, topk, {})
    return retrieved_ids


def rerank_with_cache(
    query: str,
    candidates: list[dict],
    images_dir: str | Path,
    topk: int,
    score_cache: dict[str, float],
    q_state: dict | None = None,
) -> tuple[list[str], dict[str, float], dict]:
    """
    Variant-aware re-rank: MM-R5 for images, BGE for text.

    MM-R5 is list-wise: it scores all image candidates jointly.  It is
    called on the full image pool whenever any image candidate is absent
    from score_cache.  Because build_variant() never adds image-type
    candidates, the image pool is the same across all 4 variants per
    question, so MM-R5 runs at most once per question (first variant call).

    score_cache  : {candidate_id → raw_score}
                   MM-R5 pseudo-scores for images; raw BGE cosines for text.
    q_state      : opaque dict carrying q_bge across successive variant
                   calls for the same question.

    Returns (retrieved_ids, new_scores, q_state).
      retrieved_ids : top-k quote_ids, highest fused score first.
      new_scores    : scores computed this call (not previously in cache).
    """
    assert _mm_r5_model is not None, "Call load_models() first."
    images_dir = Path(images_dir)

    img_cands = [c for c in candidates if c["type"] == "image"]
    txt_cands = [c for c in candidates if c["type"] == "text"]

    if q_state is None:
        q_state = {}

    new_scores: dict[str, float] = {}

    # ── Image candidates: MM-R5 ────────────────────────────────────────────
    img_new = [c for c in img_cands if c["quote_id"] not in score_cache]
    if img_new:
        N_img = len(img_cands)
        if N_img == 1:
            # Single-image pool: no meaningful ranking; assign 1.0 analogous
            # to Pipeline A's min-max behaviour for a single-element pool.
            new_scores[img_cands[0]["quote_id"]] = 1.0
        else:
            # List-wise: rank the FULL image pool (not just new candidates),
            # because every image's score depends on the full context.
            img_paths = [str(images_dir / c["img_path"]) for c in img_cands]
            ranked_order = _mm_r5_rerank(query, img_paths)  # 0-indexed best→worst
            for rank_0, img_idx in enumerate(ranked_order):
                rank_1 = rank_0 + 1
                pseudo = 1.0 - rank_1 / N_img  # range [0, 1)
                new_scores[img_cands[img_idx]["quote_id"]] = pseudo

    # ── Text candidates: BGE ──────────────────────────────────────────────
    txt_new = [c for c in txt_cands if c["quote_id"] not in score_cache]
    if txt_new:
        raw, q_state["q_bge"] = _score_texts_bge(
            query, txt_new, q_state.get("q_bge")
        )
        for c, s in zip(txt_new, raw):
            new_scores[c["quote_id"]] = s

    # ── Score fusion ──────────────────────────────────────────────────────
    all_scores = {**score_cache, **new_scores}
    normed: dict[str, float] = {}

    # MM-R5 pseudo-scores are already in [0, 1) — use directly as fused score.
    for c in img_cands:
        normed[c["quote_id"]] = all_scores[c["quote_id"]]

    # BGE raw cosines → min-max normalise within text pool for this variant.
    if txt_cands:
        raw_txt = [all_scores[c["quote_id"]] for c in txt_cands]
        for c, s in zip(txt_cands, _minmax(raw_txt)):
            normed[c["quote_id"]] = s

    sorted_ids = sorted(normed, key=normed.__getitem__, reverse=True)
    return sorted_ids[:topk], new_scores, q_state
