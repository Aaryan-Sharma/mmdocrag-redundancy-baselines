"""
Pipeline B generator: openbmb/MiniCPM-V-2_6.

Accepts the same evidence_items convention as Pipeline A:
  str       -> text content (text quote or text twin)
  PIL.Image -> image content

Generation settings (per spec):
  sampling=False  (greedy)
  max_new_tokens=128

MiniCPM-V-2_6 requires:
  attn_implementation='sdpa'   (explicitly required per model card)
  trust_remote_code=True

Requires Pipeline B venv (envs/requirements_b.txt).
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM-V-2_6"

_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None


def load_model(device: str = "cuda") -> None:
    global _model, _tokenizer

    print("Loading MiniCPM-V-2_6 ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation="sdpa",   # required; do not change to eager
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    print("Pipeline B generator loaded.")


def unload_model() -> None:
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache()


def generate(
    question: str,
    evidence_items: list,
    max_new_tokens: int = 128,
) -> str:
    """
    Generate a short answer given a question and ordered evidence.

    MiniCPM-V-2_6 chat() accepts a content list where images are PIL
    Image objects interleaved with strings.
    """
    assert _model is not None, "Call load_model() first."

    content: list = []
    for i, item in enumerate(evidence_items):
        if isinstance(item, Image.Image):
            content.append(item)
        else:
            content.append(f"[{i + 1}] {item}")

    content.append(
        f"\nAnswer the question concisely using the evidence above.\n"
        f"Question: {question}\nAnswer:"
    )

    msgs = [{"role": "user", "content": content}]

    # sampling=False sets generation_config={"num_beams":3,"repetition_penalty":1.2}.
    # num_beams=1 overrides via the kwargs-intersection update in chat(), giving
    # true greedy decoding — matching Pipeline A's do_sample=False behaviour.
    result = _model.chat(
        image=None,           # images are embedded in msgs content
        msgs=msgs,
        tokenizer=_tokenizer,
        sampling=False,
        max_new_tokens=max_new_tokens,
        num_beams=1,          # override default num_beams=3 → greedy
    )
    return result.strip()
