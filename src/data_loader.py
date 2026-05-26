"""
Dataset utilities: JSONL streaming, answer parsing, and the four
candidate-pool variants for MMDocRAG evaluation_20 / dev_20.
"""
import ast
import json
from pathlib import Path
from typing import Iterator

VARIANT_NAMES = frozenset(
    {"original", "gold_redundant", "negative_redundant", "mixed_redundant"}
)


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------

def stream_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(stream_jsonl(path))


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer(answer_short: str) -> str:
    """
    Normalise answer_short to a plain string.

    3.3 % of evaluation_20 items store a Python list as a string,
    e.g. "['46', '27', '64']" or "[62, 30]".  These represent
    multi-part answers (distinct required components), not alternative
    phrasings.  We join them with a space so token-overlap metrics
    measure coverage across all components at once.

    Single-element lists collapse naturally: ['Stage 5'] -> 'Stage 5'.
    """
    try:
        parsed = ast.literal_eval(answer_short)
    except (ValueError, SyntaxError):
        return answer_short.strip()
    if isinstance(parsed, list):
        return " ".join(str(x) for x in parsed)
    return str(parsed).strip()


# ---------------------------------------------------------------------------
# Text-twin derivation
# ---------------------------------------------------------------------------

def _derive_text_twin(image_quote: dict) -> dict:
    """
    Create a text-modality twin of an image quote using its VLM-generated
    img_description.  The twin shares page_id / layout_id with its source
    so document-position information is preserved.
    """
    return {
        "quote_id": f"{image_quote['quote_id']}_text",
        "type": "text",
        "text": image_quote.get("img_description", ""),
        "page_id": image_quote["page_id"],
        "layout_id": image_quote["layout_id"],
        "derived_from": image_quote["quote_id"],
    }


# ---------------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------------

def build_variant(
    item: dict,
    name: str,
) -> tuple[list[dict], set[str]]:
    """
    Return (candidates, gold_set) for the requested variant.

    candidates  — flat list of quote dicts (text_quotes + img_quotes +
                  any derived text twins), in that order.
    gold_set    — set of gold quote_ids, expanded to include twin ids
                  whenever a gold image quote receives a text twin.

    Variant semantics
    -----------------
    original          : 20 original candidates, unchanged.
    gold_redundant    : +1 twin per gold image quote  (grows gold_set).
    negative_redundant: +1 twin per non-gold image quote (gold_set unchanged).
    mixed_redundant   : +1 twin per image quote (combines both above).

    Only image quotes get twins.  Text quotes are never duplicated.
    """
    if name not in VARIANT_NAMES:
        raise ValueError(
            f"Unknown variant {name!r}. Choose from {sorted(VARIANT_NAMES)}."
        )

    text_quotes: list[dict] = item["text_quotes"]
    img_quotes: list[dict] = item["img_quotes"]
    gold_quotes: set[str] = set(item["gold_quotes"])

    candidates: list[dict] = list(text_quotes) + list(img_quotes)
    expanded_gold: set[str] = set(gold_quotes)

    if name == "original":
        return candidates, expanded_gold

    # Decide which image quotes receive a twin
    if name == "gold_redundant":
        twin_sources = [q for q in img_quotes if q["quote_id"] in gold_quotes]
    elif name == "negative_redundant":
        twin_sources = [q for q in img_quotes if q["quote_id"] not in gold_quotes]
    else:  # mixed_redundant
        twin_sources = list(img_quotes)

    for img_q in twin_sources:
        twin = _derive_text_twin(img_q)
        candidates.append(twin)
        # Expand gold set: either modality of the same evidence counts.
        if img_q["quote_id"] in gold_quotes:
            expanded_gold.add(twin["quote_id"])

    return candidates, expanded_gold
