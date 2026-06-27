"""
Rerankers (Part 2)
==================
ONE interface, many rerankers. The driver only ever calls:

    order = reranker.rerank(query, candidates)   # -> candidate indices, best-first

so switching ZipRerank <-> MM-R5 <-> a vanilla VLM is a single flag. Everything
downstream (top-k selection, answer generation, metrics) is identical.

Two reranking *styles*, both listwise over the full candidate set:

  * LogitListwiseReranker  (ZipRerank style)
      One forward pass. Lay out all candidate images labelled A,B,C,...; read the
      next-token logit of each option letter; sort descending. No generation.
      -> faithful to ZipRerank's "score by letter-token logits" recipe. Point it at
         your finetuned ZipRerank checkpoint (or base Qwen3-VL-8B for an ablation).

  * GenListwiseReranker    (MM-R5 / RankGPT style)
      Generate a (reasoning chain +) permutation string, then parse the order.
      -> MM-R5 is this with its checkpoint + CoT prompt + sequence parser.

Plugging in YOUR existing ZipRerankReranker class: just subclass BaseReranker and
return indices from your code (see CustomReranker example at the bottom), then add
it to get_reranker(). The driver never changes.

Deps: transformers, qwen-vl-utils, torch.
"""
import re
import string
from abc import ABC, abstractmethod

# Some torch/cuDNN builds fail with "No valid execution plans built" in the
# SDPA cuDNN backend (Qwen2.5-VL vision encoder). Disable just that backend;
# math/mem-efficient SDPA still work.
try:
    import torch as _torch
    _torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image


# --------------------------------------------------------------------------- #
# Candidate passed to every reranker (decoupled from MMDocIR specifics)
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    idx: int                       # stable index into the original candidate list
    text: str = ""
    _image: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            raise ValueError(f"Candidate {self.idx}: no image attached.")
        return self._image

    @property
    def has_image(self) -> bool:
        return self._image is not None


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class BaseReranker(ABC):
    name = "Base"

    @abstractmethod
    def rerank(self, query: str, candidates: List[Candidate]) -> List[int]:
        """Return candidate indices reordered best-first (a permutation of 0..n-1)."""
        ...


# --------------------------------------------------------------------------- #
# Labels A..Z, AA.. for listwise prompting / logit reading
# --------------------------------------------------------------------------- #
def _labels(n: int) -> List[str]:
    out, alpha = [], string.ascii_uppercase
    for i in range(n):
        if i < 26:
            out.append(alpha[i])
        else:
            out.append(alpha[i // 26 - 1] + alpha[i % 26])
    return out


# --------------------------------------------------------------------------- #
# Identity (baseline / debug) — keeps first-stage (ColQwen) order
# --------------------------------------------------------------------------- #
class IdentityReranker(BaseReranker):
    name = "Identity"

    def rerank(self, query, candidates):
        return list(range(len(candidates)))


# --------------------------------------------------------------------------- #
# Shared Qwen-VL loader (version-tolerant: Qwen2-VL / Qwen3-VL / others)
# --------------------------------------------------------------------------- #
def _load_vlm(model_id, device="cuda", dtype="auto", attn="sdpa"):
    import torch
    from transformers import AutoProcessor
    kwargs = dict(torch_dtype=("auto" if dtype == "auto" else getattr(torch, dtype)),
                  device_map=device, attn_implementation=attn)
    model = None
    try:  # modern unified class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _vision_info(messages):
    """Use qwen_vl_utils if present (handles Qwen image preprocessing), else fall back."""
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs
    except Exception:
        imgs = []
        for m in messages:
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "image":
                    imgs.append(c["image"])
        return (imgs or None), None


# --------------------------------------------------------------------------- #
# ZipRerank-style: logit listwise (one forward pass, no generation)
# --------------------------------------------------------------------------- #
class LogitListwiseReranker(BaseReranker):
    """
    Build one multi-image prompt with all candidates labelled A,B,C,...; take the
    model's next-token logits at the answer position; the relevance score of a
    candidate is the logit of its label token. Sort descending -> full ranking.

    This is the ZipRerank inference recipe (score-by-letter-logit). Swap in your
    finetuned checkpoint via model_id. If your ZipRerank class uses a custom head /
    two-step query-embedding extraction, override `score_candidates()` to call it.
    """
    name = "ZipRerank"

    PROMPT = (
        "You are given a query and {n} document passages, each shown as an image "
        "and labelled with a letter. Identify the passage most relevant to the query.\n"
        "Query: {query}\n"
        "Answer with the single letter of the most relevant passage."
    )

    def __init__(self, model_id, device="cuda", attn="sdpa", max_images=20):
        self.model_id = model_id
        self.max_images = max_images
        self.model, self.processor = _load_vlm(model_id, device=device, attn=attn)

    def _build_messages(self, query, candidates, labels):
        content = []
        for lab, c in zip(labels, candidates):
            content.append({"type": "text", "text": f"Passage {lab}:"})
            content.append({"type": "image", "image": c.image})
        content.append({"type": "text", "text": self.PROMPT.format(n=len(candidates), query=query)})
        return [{"role": "user", "content": content}]

    def score_candidates(self, query, candidates, labels):
        """Return a list of float scores (one per candidate) = logit of its label token."""
        import torch
        messages = self._build_messages(query, candidates, labels)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = _vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        next_logits = out.logits[0, -1, :]                      # [vocab]
        tok = self.processor.tokenizer
        scores = []
        for lab in labels:
            # score = logit of the label token (try with and without leading space)
            ids = []
            for variant in (lab, " " + lab):
                enc = tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.append(enc[0])
            scores.append(max(float(next_logits[i]) for i in ids) if ids else float("-inf"))
        return scores

    def rerank(self, query, candidates):
        n = len(candidates)
        if n <= 1:
            return list(range(n))
        cands = candidates[: self.max_images]
        labels = _labels(len(cands))
        scores = self.score_candidates(query, cands, labels)
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        # append any overflow candidates (beyond max_images) in original order
        order += list(range(len(cands), n))
        return order


# --------------------------------------------------------------------------- #
# MM-R5 / RankGPT-style: generative listwise (generate permutation, parse)
# --------------------------------------------------------------------------- #
class GenListwiseReranker(BaseReranker):
    """
    Show all candidates labelled A,B,C,...; let the model output an ordering
    (optionally preceded by a reasoning chain). Parse the permutation of labels.
    MM-R5 is this style with its checkpoint + CoT prompt.
    """
    name = "GenListwise"

    PROMPT = (
        "You are a multimodal reranker. Given a query and {n} document passages "
        "(each an image labelled with a letter), rank ALL passages from most to "
        "least relevant to the query.\n"
        "Query: {query}\n"
        "Output ONLY the ranking as a comma-separated list of letters, best first, "
        "e.g. B, A, C, ..."
    )

    def __init__(self, model_id, device="cuda", attn="sdpa", max_images=20,
                 max_new_tokens=1024, prompt=None):
        self.model_id = model_id
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens
        if prompt:
            self.PROMPT = prompt
        self.model, self.processor = _load_vlm(model_id, device=device, attn=attn)

    def _build_messages(self, query, candidates, labels):
        content = []
        for lab, c in zip(labels, candidates):
            content.append({"type": "text", "text": f"Passage {lab}:"})
            content.append({"type": "image", "image": c.image})
        content.append({"type": "text", "text": self.PROMPT.format(n=len(candidates), query=query)})
        return [{"role": "user", "content": content}]

    def _generate(self, messages):
        import torch
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = _vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    @staticmethod
    def parse_order(text, labels):
        """Extract a permutation of labels from free-form model output (robust)."""
        label_set = set(labels)
        found, seen = [], set()
        for tok in re.findall(r"[A-Z]{1,2}", text.upper()):
            if tok in label_set and tok not in seen:
                seen.add(tok); found.append(tok)
        # any labels the model omitted -> append in original order
        for lab in labels:
            if lab not in seen:
                found.append(lab)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        return [label_to_idx[lab] for lab in found]

    def rerank(self, query, candidates):
        n = len(candidates)
        if n <= 1:
            return list(range(n))
        cands = candidates[: self.max_images]
        labels = _labels(len(cands))
        out_text = self._generate(self._build_messages(query, cands, labels))
        order = self.parse_order(out_text, labels)
        order += list(range(len(cands), n))
        return order


class MMR5Reranker(BaseReranker):
    """MM-R5 (i2vec/MM-R5, Qwen2.5-VL GRPO). Exact prompt + <think>/<answer> parse."""
    name = "MM-R5"

    SYSTEM = (
        "A conversation between User and Assistant. The user asks a question, and the "
        "Assistant solves it. The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. The reasoning process and "
        "answer are enclosed within <think> </think> and <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    )
    QTEMPLATE = (
        "Please rank the following images according to their relevance to the question. "
        "Provide your response in the format: <think>your reasoning process here</think>"
        "<answer>[image_id_1, image_id_2, ...]</answer> where the numbers in the list "
        "represent the ranking order of images'id from most to least relevant. "
        "Before outputting the answer, you need to analyze each image and provide your "
        "analysis process.For example: <think>Image 1 shows the most relevant content "
        "because...</think><answer>[id_most_relevant, id_second_relevant, ...]</answer>"
        "\nThe question is: {Question}"
        "\n\nThere are {image_num} images, id from 1 to {image_num}, Image ID to image mapping:\n"
    )

    # Minimum canvas size: smaller images are padded onto white so MM-R5
    # gets enough visual tokens to analyse content (tiny strips ≈ 18×369 px
    # produce <30 patch tokens, too few for meaningful reasoning).
    MIN_SIDE = 448

    def __init__(self, model_id, device="cuda", attn="sdpa", max_images=20,
                 max_new_tokens=8192, **kw):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        self.max_images = max_images
        self.max_new_tokens = max_new_tokens
        self.last_raw = ""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation=attn,
            device_map="auto").eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id)

    @classmethod
    def _pad_to_min(cls, img: Image.Image) -> Image.Image:
        """Pad a small image onto a white canvas so the shorter side >= MIN_SIDE."""
        w, h = img.size
        if w >= cls.MIN_SIDE and h >= cls.MIN_SIDE:
            return img
        canvas_w = max(w, cls.MIN_SIDE)
        canvas_h = max(h, cls.MIN_SIDE)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        return canvas

    @staticmethod
    def _parse_ranking(decoded: str, n: int) -> List[int]:
        """
        Robustly parse 1-indexed ranking from MM-R5 output.

        Tries in order:
          1. <answer>[...] block (closing tag optional — handles truncated output;
             re.DOTALL so newlines inside the list are accepted)
          2. Last bracketed integer list anywhere in the text
          3. Integers immediately following <answer> tag (no brackets)
        Returns 0-indexed list; missing ids are NOT appended here (caller does that).
        """
        def _extract_ids(text_fragment: str, n: int) -> List[int]:
            order, seen = [], set()
            for tok in re.split(r"[\s,]+", text_fragment):
                tok = tok.strip()
                if tok.isdigit():
                    j = int(tok) - 1
                    if 0 <= j < n and j not in seen:
                        seen.add(j); order.append(j)
            return order

        # 1. <answer>[...] — closing </answer> and ] both optional (truncation safety)
        m = re.search(r"<answer>\[([\s\S]*?)(?:\]|(?=</answer>)|$)", decoded)
        if m:
            ids = _extract_ids(m.group(1), n)
            if ids:
                return ids

        # 2. Last bracketed integer list in the full text
        bracket_matches = re.findall(r"\[([0-9][0-9,\s]*?)\]", decoded)
        if bracket_matches:
            ids = _extract_ids(bracket_matches[-1], n)
            if ids:
                return ids

        # 3. Bare integers right after <answer> (no brackets at all)
        m = re.search(r"<answer>([\s\S]{1,200})", decoded)
        if m:
            ids = _extract_ids(m.group(1), n)
            if ids:
                return ids

        return []

    def rerank(self, query, candidates):
        import torch
        N = len(candidates)
        if N <= 1:
            return list(range(N))
        cands = candidates[: self.max_images]
        n = len(cands)
        content = [{"type": "text", "text": self.QTEMPLATE.format(Question=query, image_num=n)}]
        for i, c in enumerate(cands):
            content.append({"type": "text", "text": f"\nImage {i+1}: "})
            content.append({"type": "image", "image": self._pad_to_min(c.image)})
        messages = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                      max_new_tokens=self.max_new_tokens, use_cache=True)
        decoded = self.processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self.last_raw = decoded

        order = self._parse_ranking(decoded, n)
        if not order:
            print(f"  [MM-R5] parse fallback. Tail: {decoded[-300:]!r}")
        order += [i for i in range(n) if i not in order]   # missing -> identity order
        order += list(range(n, N))
        return order


# --------------------------------------------------------------------------- #
# Example: wrap your OWN existing ZipRerankReranker (from the MMDocRAG project)
# --------------------------------------------------------------------------- #
# class CustomReranker(BaseReranker):
#     name = "ZipRerank-custom"
#     def __init__(self, **kw):
#         from your_module import ZipRerankReranker      # your existing class
#         self.impl = ZipRerankReranker(**kw)
#     def rerank(self, query, candidates):
#         images = [c.image for c in candidates]
#         # adapt to your class's signature; it must return indices best-first:
#         return self.impl.rerank(query, images)


# --------------------------------------------------------------------------- #
# Registry — the swap point
# --------------------------------------------------------------------------- #
def get_reranker(name: str, **kw) -> BaseReranker:
    name = name.lower()
    if name == "identity":
        return IdentityReranker()
    if name in ("ziprerank", "zip"):
        return LogitListwiseReranker(model_id=kw.pop("model_id"), **kw)
    if name in ("mmr5", "mm-r5", "mm5"):
        return MMR5Reranker(model_id=kw.pop("model_id"), **kw)
    if name in ("vlm", "rankgpt"):
        return GenListwiseReranker(model_id=kw.pop("model_id"), **kw)
    raise ValueError(f"Unknown reranker '{name}'. Register it in get_reranker().")