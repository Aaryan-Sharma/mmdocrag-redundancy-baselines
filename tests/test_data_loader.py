"""
Tests for src/data_loader.py.

parse_answer: uses hand-crafted golden values from the data audit.
build_variant: uses the first 20 items from evaluation_20.jsonl
               (must be present at /tmp/evaluation_20.jsonl).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import build_variant, parse_answer, stream_jsonl

EVAL_JSONL = Path("/tmp/evaluation_20.jsonl")

# ---------------------------------------------------------------------------
# parse_answer
# ---------------------------------------------------------------------------

def test_parse_answer_scalar():
    assert parse_answer("0.16") == "0.16"
    assert parse_answer("  1%  ") == "1%"
    assert parse_answer("Stage 5") == "Stage 5"


def test_parse_answer_string_list_of_strings():
    # 3-element list of strings (most common list form)
    assert parse_answer("['46', '27', '64']") == "46 27 64"
    assert parse_answer("['90%', '42%', '39%']") == "90% 42% 39%"
    assert parse_answer("['yellow', 'black and white']") == "yellow black and white"


def test_parse_answer_string_list_of_ints():
    # ints in the list must be cast to str
    assert parse_answer("[62, 30]") == "62 30"


def test_parse_answer_single_element_list():
    # len-1 list collapses to plain string
    assert parse_answer("['Stage 5']") == "Stage 5"


def test_parse_answer_longer_list():
    inp = "['Radio', 'Newspaper', 'Televison', 'Internet']"
    assert parse_answer(inp) == "Radio Newspaper Televison Internet"


# ---------------------------------------------------------------------------
# build_variant — structural invariants
# ---------------------------------------------------------------------------

def _img_gold_counts(item: dict) -> tuple[int, int, int]:
    """Return (n_img, n_gold_img, n_neg_img) for one item."""
    gold = set(item["gold_quotes"])
    img_qs = item["img_quotes"]
    n_img = len(img_qs)
    n_gold_img = sum(1 for q in img_qs if q["quote_id"] in gold)
    return n_img, n_gold_img, n_img - n_gold_img


def test_build_variant_candidate_counts():
    if not EVAL_JSONL.exists():
        print(f"SKIP: {EVAL_JSONL} not found")
        return

    items = list(stream_jsonl(EVAL_JSONL))[:20]
    for item in items:
        n_txt = len(item["text_quotes"])
        n_img, n_gold_img, n_neg_img = _img_gold_counts(item)
        base = n_txt + n_img  # should always be 20

        cases = [
            ("original",           base),
            ("gold_redundant",     base + n_gold_img),
            ("negative_redundant", base + n_neg_img),
            ("mixed_redundant",    base + n_img),
        ]
        for variant, expected in cases:
            cands, _ = build_variant(item, variant)
            assert len(cands) == expected, (
                f"q_id={item['q_id']} {variant}: "
                f"expected {expected}, got {len(cands)}"
            )


def test_build_variant_gold_set_expansion():
    if not EVAL_JSONL.exists():
        print(f"SKIP: {EVAL_JSONL} not found")
        return

    items = list(stream_jsonl(EVAL_JSONL))[:20]
    for item in items:
        orig_gold = set(item["gold_quotes"])
        gold_img_ids = {
            q["quote_id"]
            for q in item["img_quotes"]
            if q["quote_id"] in orig_gold
        }

        _, gold_original = build_variant(item, "original")
        _, gold_gr = build_variant(item, "gold_redundant")
        _, gold_nr = build_variant(item, "negative_redundant")
        _, gold_mr = build_variant(item, "mixed_redundant")

        # original: gold set unchanged
        assert gold_original == orig_gold

        # gold_redundant: twin of every gold image quote added
        for qid in gold_img_ids:
            assert f"{qid}_text" in gold_gr, (
                f"q_id={item['q_id']}: twin {qid}_text missing from gold_redundant"
            )
        # non-image golds unchanged
        assert orig_gold.issubset(gold_gr)

        # negative_redundant: no new gold ids (twins come from non-gold sources)
        assert gold_nr == orig_gold

        # mixed_redundant: same expansion as gold_redundant
        assert gold_mr == gold_gr


def test_build_variant_twin_fields():
    """Derived text twins must have the correct structure."""
    if not EVAL_JSONL.exists():
        print(f"SKIP: {EVAL_JSONL} not found")
        return

    item = next(stream_jsonl(EVAL_JSONL))
    img_q = item["img_quotes"][0]

    cands, _ = build_variant(item, "mixed_redundant")
    twin_id = f"{img_q['quote_id']}_text"
    twin = next((c for c in cands if c["quote_id"] == twin_id), None)
    assert twin is not None, f"Twin {twin_id} not found in candidates"

    assert twin["type"] == "text"
    assert twin["text"] == img_q["img_description"]
    assert twin["page_id"] == img_q["page_id"]
    assert twin["layout_id"] == img_q["layout_id"]
    assert twin["derived_from"] == img_q["quote_id"]


def test_build_variant_unknown_name():
    item = {"text_quotes": [], "img_quotes": [], "gold_quotes": []}
    try:
        build_variant(item, "nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_parse_answer_scalar()
    test_parse_answer_string_list_of_strings()
    test_parse_answer_string_list_of_ints()
    test_parse_answer_single_element_list()
    test_parse_answer_longer_list()
    test_build_variant_candidate_counts()
    test_build_variant_gold_set_expansion()
    test_build_variant_twin_fields()
    test_build_variant_unknown_name()
    print("\nAll data_loader tests passed.")
