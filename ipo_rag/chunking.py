"""Chunking. `fixed` ignores structure; `section` respects paragraph blocks and chapter boundaries. Both can attach parents."""
import re

from ipo_rag.config import IngestConfig
from ipo_rag.parsing import table_text


def _split_long(text, size, overlap):
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):                                      # prefer a sentence/space boundary
            k = max(text.rfind(". ", i + size // 2, j), text.rfind(" ", i + size // 2, j))
            j = k + 1 if k > 0 else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [o for o in out if o]


def _section_children(pages, cfg):
    """Greedy paragraph packing within a chapter; the tail of each chunk overlaps into the next."""
    units, chunks = [], []                                     # unit = (chapter, page, paragraph)
    for p in pages:
        for b in p["blocks"]:
            units.append((p["chapter"], p["page"], b))
    cur, cur_ch, p0, p1 = "", None, None, None

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append({"text": cur.strip(), "chapter": cur_ch, "page_start": p0, "page_end": p1})

    for ch, pg, b in units:
        if cur and (ch != cur_ch or len(cur) + len(b) + 1 > cfg.child_chars):
            tail = cur[-cfg.child_overlap:] if ch == cur_ch and cfg.child_overlap else ""
            flush()
            cur, p0 = (tail + " " if tail else ""), pg
        if not cur.strip():
            cur_ch, p0 = ch, pg
        if len(b) > cfg.child_chars:                           # oversized paragraph: split it on its own
            for piece in _split_long(b, cfg.child_chars, cfg.child_overlap):
                chunks.append({"text": piece, "chapter": ch, "page_start": pg, "page_end": pg})
            cur = ""
            continue
        cur_ch, p1 = ch, pg
        cur = (cur + " " + b).strip() if cur.strip() else b
    flush()
    return chunks


def _fixed_children(pages, cfg):
    """Fixed windows over the whole document text; page/chapter taken from the window start."""
    stream, spans = "", []                                     # spans: (start_offset, page, chapter)
    for p in pages:
        spans.append((len(stream), p["page"], p["chapter"]))
        stream += " ".join(p["blocks"]) + " "
    chunks, i = [], 0
    import bisect
    starts = [s[0] for s in spans]
    while i < len(stream):
        j = min(len(stream), i + cfg.child_chars)
        text = stream[i:j].strip()
        if text:
            a, b = spans[bisect.bisect_right(starts, i) - 1], spans[bisect.bisect_right(starts, max(i, j - 1)) - 1]
            chunks.append({"text": text, "chapter": a[2], "page_start": a[1], "page_end": b[1]})
        if j >= len(stream):
            break
        i = j - cfg.child_overlap
    return chunks


def _attach_parents(children, cfg):
    """Group consecutive children (same chapter for `section`) into ~parent_chars parents."""
    parents, cur, size = {}, [], 0

    def close():
        nonlocal cur, size
        if cur:
            pid = f"p{len(parents)}"
            parents[pid] = {"text": " ".join(c["text"] for c in cur), "page_start": cur[0]["page_start"],
                            "page_end": cur[-1]["page_end"], "chapter": cur[0]["chapter"]}
            for c in cur:
                c["parent_id"] = pid
        cur, size = [], 0

    for c in children:
        if cur and (size + len(c["text"]) > cfg.parent_chars or (cfg.chunker == "section" and c["chapter"] != cur[0]["chapter"])):
            close()
        cur.append(c)
        size += len(c["text"])
    close()
    return parents


def build_chunks(doc, pages, tables, cfg: IngestConfig):
    """-> (chunks, parents). chunk = {id, doc, type, text, index_text, page_start, page_end, chapter, parent_id}."""
    children = (_fixed_children if cfg.chunker == "fixed" else _section_children)(pages, cfg)
    parents = _attach_parents(children, cfg) if cfg.hierarchy == "parent" else {}
    chap_of = {p["page"]: p["chapter"] for p in pages}
    chunks = [{"type": "child", "parent_id": None, **c} for c in children]
    if cfg.table_mode in ("flat", "markdown"):
        for t in tables:
            chunks.append({"type": "table", "parent_id": None, "text": table_text(t, cfg.table_mode), "chapter": chap_of.get(t["page"]),
                           "page_start": t["page"], "page_end": t["page"]})
    short = doc["company"].title().replace(" Limited", "")
    for i, c in enumerate(chunks):
        c["id"], c["doc"] = f"{doc['id']}-{i}", doc["id"]
        head = f"[{short} | {doc['doc_type'].upper()} | {(c['chapter'] or 'n/a').title()} | p.{c['page_start']}]\n" if cfg.contextual_header else ""
        c["index_text"] = head + c["text"]
    return chunks, parents
