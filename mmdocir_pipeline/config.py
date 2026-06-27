"""
Central configuration for the MMDocIR retrieval -> rerank -> generation pipeline.

Edit paths here once; every script imports from this module.
Everything that a CLI flag can override has a sensible default below.
"""
import os

# ----------------------------------------------------------------------------- #
# Dataset locations  (MMDocIR/MMDocIR_Evaluation_Dataset on HuggingFace)
# ----------------------------------------------------------------------------- #
# Download:
#   MMDocIR_annotations.jsonl   -> queries + ground truth (one line per document)
#   MMDocIR_layouts.parquet     -> 170,338 layouts ("quotes") with image_binary + text
#   MMDocIR_pages.parquet       -> 20,395 page screenshots (not needed for layout task)
DATASET_DIR   = os.environ.get("MMDOCIR_DIR", "dataset")
ANNOTATIONS   = os.path.join(DATASET_DIR, "MMDocIR_annotations.jsonl")
LAYOUTS_PARQUET = os.path.join(DATASET_DIR, "MMDocIR_layouts.parquet")
PAGES_PARQUET   = os.path.join(DATASET_DIR, "MMDocIR_pages.parquet")

# ----------------------------------------------------------------------------- #
# Retriever  (ColQwen, Faysse et al. 2025 - the first-stage retriever in ZipRerank)
# ----------------------------------------------------------------------------- #
# Canonical ColQwen2 release. To reproduce MMDocIR's reported numbers exactly,
# point this at their fine-tuned checkpoint instead (see README).
COLQWEN_MODEL = os.environ.get("COLQWEN_MODEL", "vidore/colqwen2-v1.0")

# ----------------------------------------------------------------------------- #
# Pipeline depths
# ----------------------------------------------------------------------------- #
# RETRIEVAL_TOPN = how many candidate quotes ColQwen passes to the reranker.
# ZipRerank reranks a single listwise window of ~20, so 20 is the paper default.
# Set higher (e.g. 50) only if your Part-2 reranker chunks the candidate list.
RETRIEVAL_TOPN = int(os.environ.get("RETRIEVAL_TOPN", 20))

# Modality used to represent a quote for the *retriever*.
#   "image"    -> encode the layout's cropped image_binary (visual late-interaction; default)
#   "vlm_text" -> encode the VLM/text description (text late-interaction)
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "image")

# ----------------------------------------------------------------------------- #
# Runtime
# ----------------------------------------------------------------------------- #
DEVICE        = os.environ.get("DEVICE", "cuda")
IMAGE_BS      = int(os.environ.get("IMAGE_BS", 8))    # layout-image encode batch size
QUERY_BS      = int(os.environ.get("QUERY_BS", 64))   # query encode batch size
SCORE_CHUNK   = int(os.environ.get("SCORE_CHUNK", 256))  # docs with many layouts: score in chunks
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "auto")  # "auto" | "1" | "0"

# ----------------------------------------------------------------------------- #
# Outputs
# ----------------------------------------------------------------------------- #
RETRIEVAL_DIR = os.environ.get("RETRIEVAL_DIR", "retrieval")
EMB_CACHE_DIR = os.environ.get("EMB_CACHE_DIR", "emb_cache")  # per-doc layout embeddings


def retrieval_output_path(retriever_name: str, mode: str, topn: int) -> str:
    os.makedirs(RETRIEVAL_DIR, exist_ok=True)
    return os.path.join(RETRIEVAL_DIR, f"{retriever_name}_{mode}_top{topn}.jsonl")


# ============================================================================ #
# PART 2 — reranking, answer generation, evaluation
# ============================================================================ #

# --- Reranker selection (the swap point) ---------------------------------- #
#   "ziprerank" -> logit-based listwise (one forward pass, score by option-letter logit)
#   "mmr5"      -> generative listwise (generate CoT + permutation, parse)
#   "vlm"       -> generative listwise with a vanilla Qwen-VL (RankGPT-style baseline)
#   "identity"  -> keep ColQwen order (no model; baseline / debug)
RERANKER = os.environ.get("RERANKER", "ziprerank")

# Checkpoints (override to your local paths). ZipRerank finetunes Qwen3-VL-8B-Instruct;
# point ZIPRERANK_MODEL at your finetuned checkpoint, or the base for an ablation.
ZIPRERANK_MODEL = os.environ.get("ZIPRERANK_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MMR5_MODEL      = os.environ.get("MMR5_MODEL", "i2vec/MM-R5")          # Qwen2.5-VL-7B based
VLM_RERANK_MODEL = os.environ.get("VLM_RERANK_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

# --- Answer generator ----------------------------------------------------- #
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
#   multimodal -> feed top-k candidate IMAGES to the generator
#   pure-text  -> feed top-k candidate TEXTS (vlm_text/text)
#   hybrid     -> feed both image AND its text
GENERATION_MODE = os.environ.get("GENERATION_MODE", "multimodal")

# --- Depths --------------------------------------------------------------- #
# k = how many reranked quotes are handed to the generator (the paper uses ~ up to 20).
TOPK = int(os.environ.get("TOPK", 5))
# Max images per VLM call (rerank window / generator context guard).
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 20))

# --- Generation decoding -------------------------------------------------- #
GEN_MAX_NEW_TOKENS    = int(os.environ.get("GEN_MAX_NEW_TOKENS", 512))
RERANK_MAX_NEW_TOKENS = int(os.environ.get("RERANK_MAX_NEW_TOKENS", 1024))  # MM-R5 CoT needs room

# --- Outputs -------------------------------------------------------------- #
RESPONSE_DIR = os.environ.get("RESPONSE_DIR", "response")
EVAL_DIR     = os.environ.get("EVAL_DIR", "eval")


def response_output_path(reranker: str, gen_mode: str, topk: int) -> str:
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    return os.path.join(RESPONSE_DIR, f"{reranker}_{gen_mode}_top{topk}_response.jsonl")


def eval_output_path(reranker: str, gen_mode: str, topk: int) -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    return os.path.join(EVAL_DIR, f"{reranker}_{gen_mode}_top{topk}_eval.json")
