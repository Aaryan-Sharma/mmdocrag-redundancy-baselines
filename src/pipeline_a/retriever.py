"""
Pipeline A retriever: ColPali + BGE hybrid.

Image candidates -> vidore/colpali-v1.3-hf (MaxSim, multi-vector).
Text candidates  -> BAAI/bge-large-en-v1.5 (cosine, sentence-transformers).

Score fusion (per question):
  1. Min-max normalise ColPali scores across image candidates to [0, 1].
  2. Min-max normalise BGE scores across text candidates to [0, 1].
  3. Merge both pools; sort by normalised score; return top-k ids.

When a pool has only one candidate the normalised score is 1.0 (no
meaningful range), which is correct — it will compete with whatever
the other pool produces.

Requires Pipeline A venv (envs/requirements_a.txt).

Note: uses colpali-engine (not HF ColPaliForRetrieval) because the
colpali-v1.3-hf checkpoint was saved with transformers==4.48.0.dev0
and HF's ColPaliForRetrieval in transformers>=4.49 has incompatible
key names. colpali-engine carries its own _checkpoint_conversion_mapping
that remaps keys correctly regardless of transformers version.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Module-level model singletons (loaded once per process)
# ---------------------------------------------------------------------------

_colpali_model: ColPali | None = None
_colpali_processor: ColPaliProcessor | None = None
_bge_model: SentenceTransformer | None = None


def load_models(device: str = "cuda") -> None:
    """Load ColPali and BGE models into module-level singletons."""
    global _colpali_model, _colpali_processor, _bge_model

    _COLPALI_KEY_MAPPING = {
        # checkpoint (saved with old GemmaForCausalLM wrapper) has:
        #   vlm.language_model.model.*  -> strip the .model. wrapper
        r"^vlm\.language_model\.model\.": "model.model.language_model.",
        # vision_tower is SiglipVisionModel; keep .vision_model. intact
        r"^vlm\.vision_tower\.": "model.model.vision_tower.",
        r"^vlm\.multi_modal_projector\.": "model.model.multi_modal_projector.",
        r"^embedding_proj_layer\.": "custom_text_proj.",
    }
    print("Loading ColPali v1.3-hf (via colpali-engine) ...")
    _colpali_model = ColPali.from_pretrained(
        "vidore/colpali-v1.3-hf",
        torch_dtype=torch.bfloat16,
        device_map=device,
        key_mapping=_COLPALI_KEY_MAPPING,
    ).eval()
    _colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")

    print("Loading BGE-large-en-v1.5 ...")
    _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Pipeline A retriever models loaded.")


def unload_models() -> None:
    """Release GPU memory before loading the generator."""
    global _colpali_model, _colpali_processor, _bge_model
    _colpali_model = None
    _colpali_processor = None
    _bge_model = None
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _minmax(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _score_images_colpali(
    query: str,
    img_candidates: list[dict],
    images_dir: Path,
    q_colpali: torch.Tensor | None = None,
) -> tuple[list[float], torch.Tensor]:
    """
    Return (raw_scores, q_colpali).  Encodes query if q_colpali is None.
    """
    assert _colpali_model is not None and _colpali_processor is not None

    if q_colpali is None:
        batch_q = _colpali_processor.process_queries([query]).to(_colpali_model.device)
        with torch.no_grad():
            q_colpali = _colpali_model(**batch_q)

    images = [
        Image.open(images_dir / c["img_path"]).convert("RGB")
        for c in img_candidates
    ]
    batch_images = _colpali_processor.process_images(images).to(_colpali_model.device)
    with torch.no_grad():
        img_embs = _colpali_model(**batch_images)

    scores = _colpali_processor.score_multi_vector(q_colpali, img_embs)
    return scores[0].float().cpu().tolist(), q_colpali


def _score_texts_bge(
    query: str,
    txt_candidates: list[dict],
    q_bge: np.ndarray | None = None,
) -> tuple[list[float], np.ndarray]:
    """Return (raw_scores, q_bge).  Encodes query if q_bge is None."""
    assert _bge_model is not None

    if q_bge is None:
        q_bge = _bge_model.encode([query], normalize_embeddings=True)

    texts = [c["text"] for c in txt_candidates]
    t_embs = _bge_model.encode(texts, normalize_embeddings=True)
    scores = (q_bge @ t_embs.T)[0]
    return scores.tolist(), q_bge


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

    score_cache stores raw ColPali/BGE scores (before min-max normalization).
    Normalization is applied at call time within the variant's candidate pool,
    so results are always correct for the given variant regardless of what
    other variants have been processed.

    score_cache : {candidate_id → raw_score} for known candidates of this
                  question.
    q_state     : opaque dict carrying query embeddings (q_colpali, q_bge)
                  across successive variant calls for the same question.

    Returns (retrieved_ids, new_scores, q_state).
    """
    assert _colpali_model is not None, "Call load_models() first."
    images_dir = Path(images_dir)

    img_cands = [c for c in candidates if c["type"] == "image"]
    txt_cands = [c for c in candidates if c["type"] == "text"]

    img_new = [c for c in img_cands if c["quote_id"] not in score_cache]
    txt_new = [c for c in txt_cands if c["quote_id"] not in score_cache]

    if q_state is None:
        q_state = {}

    new_scores: dict[str, float] = {}

    # Score new image candidates (ColPali)
    if img_new:
        raw, q_state["q_colpali"] = _score_images_colpali(
            query, img_new, images_dir, q_state.get("q_colpali")
        )
        for c, s in zip(img_new, raw):
            new_scores[c["quote_id"]] = s

    # Score new text candidates (BGE)
    if txt_new:
        raw, q_state["q_bge"] = _score_texts_bge(
            query, txt_new, q_state.get("q_bge")
        )
        for c, s in zip(txt_new, raw):
            new_scores[c["quote_id"]] = s

    # Apply min-max normalization within modality for this variant's pool
    all_scores = {**score_cache, **new_scores}
    normed: dict[str, float] = {}

    if img_cands:
        raw_img = [all_scores[c["quote_id"]] for c in img_cands]
        for c, s in zip(img_cands, _minmax(raw_img)):
            normed[c["quote_id"]] = s

    if txt_cands:
        raw_txt = [all_scores[c["quote_id"]] for c in txt_cands]
        for c, s in zip(txt_cands, _minmax(raw_txt)):
            normed[c["quote_id"]] = s

    # normed only contains "image" and "text" candidates; "table" etc. are
    # silently excluded to match the original rerank() behaviour.
    sorted_ids = sorted(normed, key=normed.__getitem__, reverse=True)
    return sorted_ids[:topk], new_scores, q_state
