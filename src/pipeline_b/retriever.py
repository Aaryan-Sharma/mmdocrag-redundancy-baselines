"""
Pipeline B retriever: VisRAG-Ret.

Text candidates (including derived twins) and image candidates are both
scored by the same model via its unified encode() interface:
  - list[str]       -> text-mode embedding
  - list[PIL.Image] -> image-mode embedding

Similarity: cosine (L2-normalised embeddings, dot product).

VisRAG-Ret was trained on full page images; the MMDocRAG image quotes
are element-level crops.  This is a known distribution shift noted in
the study design; it affects all variants uniformly so comparisons
across variants remain valid.

Query prefix (from HF model card):
  "Represent this query for retrieving relevant documents: "

Requires Pipeline B venv (envs/requirements_b.txt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

QUERY_PREFIX = "Represent this query for retrieving relevant documents: "
MODEL_ID = "openbmb/VisRAG-Ret"

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None


def load_models(device: str = "cuda") -> None:
    global _model, _tokenizer

    print("Loading VisRAG-Ret ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print("Pipeline B retriever model loaded.")


def unload_models() -> None:
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Encoding helpers (verbatim from HF model card)
# ---------------------------------------------------------------------------

def _weighted_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    return s / d


@torch.no_grad()
def _encode(items: list) -> np.ndarray:
    """
    Encode a homogeneous batch of strings or PIL Images.
    Returns L2-normalised float32 embeddings of shape (n, d).
    """
    assert _model is not None and _tokenizer is not None

    if isinstance(items[0], str):
        inputs = {
            "text": items,
            "image": [None] * len(items),
            "tokenizer": _tokenizer,
        }
    else:
        inputs = {
            "text": [""] * len(items),
            "image": items,
            "tokenizer": _tokenizer,
        }

    outputs = _model(**inputs)
    reps = _weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
    return F.normalize(reps, p=2, dim=1).detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# Public interface
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
    Variant-aware re-rank using cached per-candidate raw scores.

    score_cache : {candidate_id → raw_score} for known candidates of this
                  question.  Candidates absent from cache are scored fresh.
    q_state     : opaque dict carrying the query embedding across successive
                  variant calls for the same question.  Pass None on the
                  first call; pass the returned dict on subsequent calls.

    Returns
    -------
    retrieved_ids : top-k candidate IDs in descending score order
    new_scores    : dict of newly computed scores (caller should persist)
    q_state       : updated state dict (pass to next variant for same q)
    """
    assert _model is not None, "Call load_models() first."
    images_dir = Path(images_dir)

    img_new = [c for c in candidates if c["type"] == "image"
               and c["quote_id"] not in score_cache]
    txt_new = [c for c in candidates if c["type"] == "text"
               and c["quote_id"] not in score_cache]

    new_scores: dict[str, float] = {}

    if img_new or txt_new:
        if q_state is None:
            q_state = {}
        if "q_emb" not in q_state:
            q_state["q_emb"] = _encode([QUERY_PREFIX + query])
        q_emb: np.ndarray = q_state["q_emb"]

        if img_new:
            images = [
                Image.open(images_dir / c["img_path"]).convert("RGB")
                for c in img_new
            ]
            img_embs = _encode(images)
            for c, s in zip(img_new, (q_emb @ img_embs.T)[0].tolist()):
                new_scores[c["quote_id"]] = s

        if txt_new:
            txt_embs = _encode([c["text"] for c in txt_new])
            for c, s in zip(txt_new, (q_emb @ txt_embs.T)[0].tolist()):
                new_scores[c["quote_id"]] = s
    else:
        if q_state is None:
            q_state = {}

    all_scores = {**score_cache, **new_scores}
    # Skip candidates whose type is neither "image" nor "text" (e.g. "table").
    # The original rerank() silently dropped these; maintain that behaviour.
    variant_scores = {
        c["quote_id"]: all_scores[c["quote_id"]]
        for c in candidates
        if c["quote_id"] in all_scores
    }
    sorted_ids = sorted(variant_scores, key=variant_scores.__getitem__, reverse=True)
    return sorted_ids[:topk], new_scores, q_state
