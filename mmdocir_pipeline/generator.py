"""
Answer generation (Part 2)
=========================
Given a query and the top-k reranked quotes, generate the final answer with a
VLM (default Qwen2-VL-7B-Instruct). Three modes, matching the MMDocRAG setup:

  multimodal : feed the top-k candidate IMAGES
  pure-text  : feed the top-k candidate TEXTS (vlm_text / text)
  hybrid     : feed BOTH the image and its text for each quote

Deps: transformers, qwen-vl-utils, torch.
"""
from typing import List

from rerankers import Candidate, _load_vlm, _vision_info


SYSTEM = (
    "You answer questions about a document using only the provided passages. "
    "Be concise and factual. If the passages are insufficient, say so."
)


class AnswerGenerator:
    def __init__(self, model_id, device="cuda", attn="sdpa", max_new_tokens=512):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = _load_vlm(model_id, device=device, attn=attn)

    def _build_messages(self, query: str, quotes: List[Candidate], mode: str):
        content = []
        content.append({"type": "text",
                        "text": f"Question: {query}\n\nUse these passages to answer:"})
        for n, q in enumerate(quotes, 1):
            content.append({"type": "text", "text": f"[Passage {n}]"})
            if mode in ("multimodal", "hybrid") and q.has_image:
                content.append({"type": "image", "image": q.image})
            if mode in ("pure-text", "hybrid"):
                content.append({"type": "text", "text": q.text or "(no text)"})
        content.append({"type": "text",
                        "text": "\nAnswer the question based on the passages above."})
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]

    def generate(self, query: str, quotes: List[Candidate], mode: str = "multimodal") -> str:
        import torch
        messages = self._build_messages(query, quotes, mode)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = _vision_info(messages) if mode != "pure-text" else (None, None)
        inputs = self.processor(text=[text], images=image_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
