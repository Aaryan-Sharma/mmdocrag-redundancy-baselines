"""Pipeline C generator: thin re-export from pipeline_a.generator.

Pipeline C uses the same Qwen2.5-VL-7B-Instruct generator as Pipeline A,
with identical greedy-decoding settings (do_sample=False, num_beams=1).
"""
from pipeline_a.generator import generate, load_model, unload_model  # noqa: F401
