"""
Retrievers
==========
A small swappable interface so the first-stage retriever can be changed without
touching the driver (mirrors the reranker-swap design used in Part 2).

Default: ColQwen2 (Faysse et al. 2025) — the multi-vector late-interaction
retriever used as ZipRerank's first stage. Uses the official `colpali_engine`
API (model(**process_images(...)) -> multi-vector; score_multi_vector -> MaxSim),
which is logic-identical to MMDocIR's colbert_score.

    pip install colpali-engine

To reproduce MMDocIR's published numbers exactly, set COLQWEN_MODEL to their
fine-tuned checkpoint (MMDocIR/MMDocIR_Retrievers/colqwen2-v1.0). The interface
is unchanged.
"""
from abc import ABC, abstractmethod
from typing import List

import torch
from PIL import Image
from tqdm import tqdm


class BaseRetriever(ABC):
    """Encode queries and quotes, then score query-vs-quotes."""
    name = "Base"

    @abstractmethod
    def embed_queries(self, queries: List[str]): ...

    @abstractmethod
    def embed_image_quotes(self, images: List[Image.Image]): ...

    @abstractmethod
    def embed_text_quotes(self, texts: List[str]): ...

    @abstractmethod
    def score(self, query_embs, quote_embs):
        """Return a score matrix [n_queries, n_quotes] (CPU tensor or ndarray)."""
        ...


class ColQwenRetriever(BaseRetriever):
    name = "ColQwen"

    def __init__(self, model_name="vidore/colqwen2-v1.0", device="cuda",
                 image_bs=8, query_bs=64, use_flash_attn="auto"):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        if use_flash_attn == "auto":
            try:
                from transformers.utils.import_utils import is_flash_attn_2_available
                attn = "flash_attention_2" if is_flash_attn_2_available() else None
            except Exception:
                attn = None
        else:
            attn = "flash_attention_2" if str(use_flash_attn) == "1" else None

        self.device = device
        self.image_bs = image_bs
        self.query_bs = query_bs
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        print(f"[ColQwenRetriever] loaded '{model_name}' (attn={attn}) on {device}")

    # -- encoding ---------------------------------------------------------- #
    @torch.no_grad()
    def _embed(self, items, kind: str):
        """kind: 'image' | 'query' | 'text'. Returns list of CPU multi-vector tensors."""
        bs = self.image_bs if kind == "image" else self.query_bs
        out = []
        for i in tqdm(range(0, len(items), bs), desc=f"[ColQwen] embed {kind}", leave=False):
            batch_items = items[i:i + bs]
            if kind == "image":
                batch = self.processor.process_images(batch_items)
            else:  # query OR text quote — both processed as queries (text-only) by ColQwen
                batch = self.processor.process_queries(batch_items)
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            emb = self.model(**batch)               # [B, L, D] (padded per batch)
            out.extend(list(torch.unbind(emb.to("cpu").float())))
        return out

    def embed_queries(self, queries):       return self._embed(queries, "query")
    def embed_image_quotes(self, images):   return self._embed(images, "image")
    def embed_text_quotes(self, texts):     return self._embed(texts, "text")

    # -- scoring (ColBERT MaxSim) ------------------------------------------ #
    @torch.no_grad()
    def score(self, query_embs, quote_embs):
        """
        query_embs / quote_embs: list of [L_i, D] CPU tensors.
        Returns [n_queries, n_quotes] CPU tensor. Equivalent to MMDocIR colbert_score
        (sum over query tokens of max over quote tokens).
        """
        return self.processor.score_multi_vector(query_embs, quote_embs)


def get_retriever(name: str, **kwargs) -> BaseRetriever:
    name = name.lower()
    if name in ("colqwen", "colqwen2"):
        return ColQwenRetriever(**kwargs)
    raise ValueError(f"Unknown retriever '{name}'. Add it to retrievers.get_retriever().")
