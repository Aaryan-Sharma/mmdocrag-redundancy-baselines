# mmdocrag_redundancy_v1

Frozen dataset artifacts for the MMDocRAG redundancy study.  
**Do not modify these files.**  Any change to variant construction logic must produce a new versioned directory.

---

## Directory layout

```
mmdocrag_redundancy_v1/
├── data/
│   ├── original_eval.jsonl
│   ├── gold_redundant_eval.jsonl
│   ├── negative_redundant_eval.jsonl
│   ├── mixed_redundant_eval.jsonl
│   └── dev/
│       ├── original_dev.jsonl
│       ├── gold_redundant_dev.jsonl
│       ├── negative_redundant_dev.jsonl
│       └── mixed_redundant_dev.jsonl
├── build_variant.py        # standalone copy of build logic
├── checksums.txt           # md5 of each JSONL
└── README.md               # this file
```

**Eval split**: 2000 items from `evaluation_20.jsonl`.  
**Dev split**: 2055 items from `dev_20.jsonl` (for debugging/fast iteration).

---

## Row schema

Each JSONL line is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `q_id` | int | Question ID (unique within split; not unique across splits) |
| `question` | str | Question text |
| `candidates` | list[dict] | Ordered candidate pool for this variant (see below) |
| `gold_set` | list[str] | Sorted list of gold `quote_id`s (variant-expanded) |
| `answer_ref` | str | Parsed/normalised answer (via `parse_answer`) |
| `answer_short_raw` | str | Raw `answer_short` from source (may be a Python-list string) |
| `answer_interleaved` | str | Long-form interleaved answer from source |

### Candidate dict fields

Text quotes and text twins:
```json
{"quote_id": "text3", "type": "text", "text": "...", "page_id": 4, "layout_id": 37}
```

Image quotes:
```json
{"quote_id": "image2", "type": "image", "img_path": "images/doc_image2.jpg",
 "img_description": "...", "page_id": 2, "layout_id": 15}
```

Derived text twins (added by redundancy variants):
```json
{"quote_id": "image2_text", "type": "text", "text": "...",
 "page_id": 2, "layout_id": 15, "derived_from": "image2"}
```

---

## Construction rules

1. **Candidate ordering**: `text_quotes` first, then `img_quotes`, then derived twins (appended in img_quotes order).
2. **Text quotes are never duplicated** — only `type=image` quotes receive twins.
3. **Twin content** = `img_description` field of the source image quote.
4. **Twin `quote_id`** = `{source_quote_id}_text` (e.g. `"image4"` → `"image4_text"`).
5. **Twin inherits** `page_id` and `layout_id` from its source image quote.
6. **gold_set expansion**: twin `quote_id` is added to `gold_set` iff the source image quote was already gold. Non-gold twins do not enter `gold_set`.
7. **`answer_ref`** is computed by `parse_answer(answer_short_raw)`: Python-list strings like `"['46', '27', '64']"` are parsed and components joined with a space (`"46 27 64"`). Plain strings are returned stripped.
8. **`gold_set`** is stored as a sorted list (not a set) for JSON serialisability.

---

## Variant definitions

| Variant | Candidates | gold_set grows? | Avg pool (eval) |
|---|---|---|---|
| `original` | 20 base (text + image) | No | ~20 |
| `gold_redundant` | +1 twin per **gold** image quote | Yes | ~21.5 |
| `negative_redundant` | +1 twin per **non-gold** image quote | No | ~26 |
| `mixed_redundant` | +1 twin per **every** image quote | Yes | ~27.7 |

Pool sizes vary per item because source docs have different numbers of text vs image quotes.  The max cap is 28 (20 base + up to 8 twins when all images are non-gold).

---

## Pool-size and gold-set sanity (eval split)

| Variant | pool avg | pool min | pool max | gold avg | gold min | gold max |
|---|---|---|---|---|---|---|
| original | 19.9 | 12 | 20 | 2.9 | 1 | 12 |
| gold_redundant | 21.5 | 14 | 28 | 4.5 | 1 | 16 |
| negative_redundant | 26.1 | 13 | 28 | 2.9 | 1 | 12 |
| mixed_redundant | 27.7 | 14 | 28 | 4.5 | 1 | 16 |

---

## Answer parsing

`answer_short_raw` is the verbatim `answer_short` string from the source dataset.
Approximately 3.3% of eval items store a Python list literal, e.g. `"['46', '27', '64']"`.
These represent **multi-part answers** (distinct required components), not alternatives.

`answer_ref = parse_answer(answer_short_raw)` normalises them:
- `"['46', '27', '64']"` → `"46 27 64"`
- `"0.16"` → `"0.16"`
- `"['Stage 5']"` → `"Stage 5"`

Use `answer_ref` for token-overlap metrics (F1, recall). Keep `answer_short_raw` if you need to inspect the original.

---

## Usage

```python
import json

with open("mmdocrag_redundancy_v1/data/gold_redundant_eval.jsonl") as f:
    for line in f:
        row = json.loads(line)
        # row["candidates"]  — list of candidate dicts
        # row["gold_set"]    — list of gold quote_ids
        # row["answer_ref"]  — normalised reference answer
```

To re-derive from scratch (useful for verification):
```python
from build_variant import build_variant, parse_answer
import json

with open("/path/to/evaluation_20.jsonl") as f:
    for line in f:
        item = json.loads(line)
        candidates, gold_set = build_variant(item, "gold_redundant")
        answer_ref = parse_answer(item["answer_short"])
```

---

## Integrity verification

```bash
cd mmdocrag_redundancy_v1
md5sum -c checksums.txt
```

All 8 lines should report `OK`.
