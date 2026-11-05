"""PDF -> pages (reading-order text blocks + chapter label) and tables. Tables are cached: detection costs minutes per filing."""
import hashlib
import json
import re
from pathlib import Path

import fitz

import os

from ipo_rag.structure import label_pages, label_pages_sec

CACHE = Path(__file__).resolve().parent.parent / "indexes" / "_parse_cache"


def _clean(t):
    return re.sub(r"\s+", " ", t.replace("­", "")).strip()


def load_pages(pdf, max_pages=None):
    """[{page, text, blocks:[str], chapter}] — blocks are paragraphs in reading order (page-number-only blocks dropped)."""
    doc = fitz.open(pdf)
    n = min(len(doc), max_pages) if max_pages else len(doc)
    pages, texts = [], []
    for i in range(n):
        raw = doc[i].get_text()
        blocks = [_clean(b[4]) for b in doc[i].get_text("blocks", sort=True) if b[6] == 0]
        blocks = [b for b in blocks if len(b) > 2 and not re.fullmatch(r"[\dIVXLCivxlc]{1,5}", b)]
        pages.append({"page": i + 1, "text": raw, "blocks": blocks})
        texts.append(raw)
    labels = label_pages_sec(texts) if os.environ.get("RAG_STRUCTURE", "sebi") == "sec" else label_pages(texts)   # sec = US 10-K/10-Q layout
    for p, ch in zip(pages, labels):
        p["chapter"] = ch
    return pages


def extract_tables(pdf, max_pages=None):
    """[{page, rows:[[cell]], caption}] via PyMuPDF find_tables; cached per file hash."""
    pdf = Path(pdf)
    key = hashlib.sha1(f"{pdf.name}-{pdf.stat().st_size}-{max_pages}".encode()).hexdigest()[:12]
    cp = CACHE / f"{key}.tables.json"
    if cp.exists():
        return json.load(open(cp))
    doc = fitz.open(pdf)
    out = []
    for i in range(min(len(doc), max_pages or len(doc))):
        page = doc[i]
        try:
            found = page.find_tables()
        except Exception:
            continue
        for t in found:
            rows = [[_clean(c or "") for c in r] for r in (t.extract() or [])]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2 or max(len(r) for r in rows) < 2 or sum(len(c) for r in rows for c in r) < 80:
                continue
            cap = None
            for b in page.get_text("blocks", sort=True):          # nearest short block above the table
                if b[3] <= t.bbox[1] + 2 and t.bbox[1] - b[3] < 60 and b[6] == 0:
                    txt = _clean(b[4])
                    if 5 < len(txt) < 140:
                        cap = txt
            out.append({"page": i + 1, "rows": rows, "caption": cap})
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cp, "w"))
    return out


def table_text(tbl, mode):
    """Render one table. flat = 'a | b' rows (current pipeline). markdown = captioned markdown table."""
    rows = tbl["rows"]
    if mode == "flat":
        return "\n".join(" | ".join(r) for r in rows)
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return (f"Table: {tbl['caption']}\n" if tbl.get("caption") else "") + "\n".join(md)
