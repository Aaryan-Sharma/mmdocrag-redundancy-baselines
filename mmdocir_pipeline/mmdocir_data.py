"""
MMDocIR data layer
==================
Turns the raw MMDocIR evaluation files into clean, pipeline-ready objects.

Raw inputs
----------
MMDocIR_annotations.jsonl : one JSON line per document
    {
      "doc_name": str, "domain": str,
      "page_indices":   [start_row, end_row],   # inclusive rows into pages.parquet
      "layout_indices": [start_row, end_row],   # inclusive rows into layouts.parquet
      "questions": [ {"Q","A","type","page_id","layout_mapping"}, ... ]
    }
    NOTE: the HuggingFace card prints this key as "layout_indinces" (typo). The
    actual file + official encode.py use "layout_indices"; we read that and fall
    back to the typo just in case.

MMDocIR_layouts.parquet : one row per layout ("quote")
    doc_name, domain, type, layout_id, page_id, image_path, image_binary,
    text (text/equation only), ocr_text & vlm_text (table/figure only),
    bbox [x1,y1,x2,y2], page_size [w,h]

Concepts
--------
A "quote" = one layout. A document's quotes are the parquet rows
    layouts_df.iloc[start : end+1]   (slice is inclusive of end, per official search.py)
The position of a quote within that slice is its `local_idx`, which is what the
score vector and the official recall_layout() are indexed by.
"""
import io
import json
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from PIL import Image

from mmdocir_metrics import candidate_gold_overlap


# --------------------------------------------------------------------------- #
# Quote (layout) entity
# --------------------------------------------------------------------------- #
@dataclass
class Quote:
    local_idx: int            # position within the document's layout slice (parallel to scores)
    row_index: int            # absolute row in layouts.parquet (used to rehydrate image in Part 2)
    layout_id: int
    page_id: int
    type: str                 # text | equation | table | image | figure | ...
    text: str                 # textual representation (see text_for_layout)
    bbox: List[float]         # [x1, y1, x2, y2]
    page_size: List[float]    # [w, h]
    image_path: str
    _image_binary: Optional[bytes] = field(default=None, repr=False)

    def image(self) -> Image.Image:
        """Decode the cropped layout image (RGB)."""
        if self._image_binary is None:
            raise ValueError(f"Quote {self.layout_id}: image_binary not loaded.")
        return Image.open(io.BytesIO(self._image_binary)).convert("RGB")

    def index_tuple(self):
        """(row_index, page_id, x1, y1, x2, y2) — the format recall_layout expects."""
        return (self.row_index, self.page_id, *self.bbox)

    def to_record(self):
        """Serialisable metadata (no raw image bytes) for the retrieval handoff file."""
        return {
            "local_idx": self.local_idx,
            "row_index": self.row_index,
            "layout_id": self.layout_id,
            "page_id": self.page_id,
            "type": self.type,
            "text": self.text,
            "bbox": self.bbox,
            "page_size": self.page_size,
            "image_path": self.image_path,
        }


@dataclass
class Query:
    qid: int                  # global question id (annotation order; matches official q_count)
    doc_name: str
    domain: str
    question: str
    answer: str
    question_type: str
    gold_page_id: List[int]                 # gold pages
    layout_mapping: List[dict]              # gold regions [{page, page_size, bbox}, ...]
    layout_indices: List[int]              # [start_row, end_row] inclusive, into layouts.parquet


# --------------------------------------------------------------------------- #
# Text selection — mirror official encode.py get_layouts()
# --------------------------------------------------------------------------- #
def text_for_layout(row) -> str:
    """
    Official rule (encode.py): image-like layouts use the VLM/OCR description,
    everything else uses raw `text`. We add a safe fallback chain so unexpected
    type vocab never yields an empty string when *some* text exists.
    """
    ltype = (row["type"] or "").lower()
    candidates = []
    if ltype in ("table", "image", "figure"):
        candidates = [row.get("vlm_text"), row.get("ocr_text"), row.get("text")]
    else:  # text, equation, ...
        candidates = [row.get("text"), row.get("vlm_text"), row.get("ocr_text")]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return ""


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_layouts_parquet(path: str) -> pd.DataFrame:
    """Load layouts parquet once. Reset index so positional .iloc == row_index."""
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)


def _layout_index_range(doc: dict) -> List[int]:
    rng = doc.get("layout_indices", doc.get("layout_indinces"))  # tolerate HF-card typo
    if rng is None:
        raise KeyError("Annotation is missing 'layout_indices'.")
    return list(rng)


def load_queries(annotations_path: str) -> List[Query]:
    """Expand annotations into a flat list of Query objects, in annotation order."""
    queries: List[Query] = []
    qid = 0
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            layout_indices = _layout_index_range(doc)
            for qa in doc["questions"]:
                queries.append(Query(
                    qid=qid,
                    doc_name=doc["doc_name"],
                    domain=doc["domain"],
                    question=qa["Q"],
                    answer=qa.get("A", ""),
                    question_type=qa.get("type", ""),
                    gold_page_id=qa.get("page_id", []),
                    layout_mapping=qa.get("layout_mapping", []),
                    layout_indices=layout_indices,
                ))
                qid += 1
    return queries


def build_quotes(layouts_df: pd.DataFrame, layout_indices: List[int],
                 load_images: bool = True) -> List[Quote]:
    """
    Slice a document's layouts and build Quote entities.
    `layout_indices` = [start_row, end_row] inclusive (official convention).
    """
    start, end = int(layout_indices[0]), int(layout_indices[1])
    sub = layouts_df.iloc[start:end + 1]
    quotes: List[Quote] = []
    for local_idx, (row_index, row) in enumerate(sub.iterrows()):
        quotes.append(Quote(
            local_idx=local_idx,
            row_index=int(row_index),
            layout_id=int(row["layout_id"]),
            page_id=int(row["page_id"]),
            type=str(row["type"]),
            text=text_for_layout(row),
            bbox=[float(x) for x in row["bbox"]],
            page_size=[float(x) for x in row["page_size"]],
            image_path=str(row.get("image_path", "")),
            _image_binary=(row["image_binary"] if load_images else None),
        ))
    return quotes


def label_gold(quotes: List[Quote], layout_mapping: List[dict]) -> List[float]:
    """Per-candidate gold overlap area (>0 => touches a gold region)."""
    return [candidate_gold_overlap(q.page_id, q.bbox, layout_mapping) for q in quotes]


# --------------------------------------------------------------------------- #
# Grouping — queries share a document; encode each document's layouts once
# --------------------------------------------------------------------------- #
def group_queries_by_document(queries: List[Query]):
    """
    Returns ordered dict-like: doc_name -> {"layout_indices": [...], "queries": [Query...]}
    All queries of a document share the same layout slice, so we encode it once.
    """
    grouped = {}
    for q in queries:
        g = grouped.setdefault(q.doc_name, {"layout_indices": q.layout_indices, "queries": []})
        g["queries"].append(q)
    return grouped
