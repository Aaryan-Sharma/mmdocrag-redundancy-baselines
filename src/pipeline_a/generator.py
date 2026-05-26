"""
Pipeline A generator: Qwen/Qwen2.5-VL-7B-Instruct.

Accepts an ordered list of evidence items (str for text/twin candidates,
PIL.Image for image candidates) plus the question string.
Returns the raw generated string.

Generation settings (per spec):
  temperature = 0  (greedy, implemented as do_sample=False)
  max_new_tokens = 128

Requires Pipeline A venv (envs/requirements_a.txt).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "qwen-vl-utils is required for Pipeline A generator. "
        "Install it: pip install qwen-vl-utils"
    ) from e

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

_model: Qwen2_5_VLForConditionalGeneration | None = None
_processor: AutoProcessor | None = None


def load_model(device: str = "cuda") -> None:
    global _model, _processor

    print("Loading Qwen2.5-VL-7B-Instruct ...")
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Pipeline A generator loaded.")


def unload_model() -> None:
    global _model, _processor
    _model = None
    _processor = None
    torch.cuda.empty_cache()


def generate(
    question: str,
    evidence_items: list,
    max_new_tokens: int = 128,
) -> str:
    """
    Generate a short answer given a question and ordered evidence.

    evidence_items : list where each element is either
                     - str        (text or twin candidate content)
                     - PIL.Image  (image candidate)
                     Items appear in retrieval-rank order (highest first).
    """
    assert _model is not None, "Call load_model() first."

    content: list[dict] = []
    for i, item in enumerate(evidence_items):
        if isinstance(item, Image.Image):
            content.append({"type": "image", "image": item})
        else:
            content.append({"type": "text", "text": f"[{i + 1}] {item}"})

    content.append({
        "type": "text",
        "text": f"\nAnswer the question concisely using the evidence above.\n"
                f"Question: {question}\nAnswer:",
    })

    messages = [{"role": "user", "content": content}]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # greedy = temperature 0
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return _processor.decode(new_ids, skip_special_tokens=True).strip()
