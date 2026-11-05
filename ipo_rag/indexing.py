"""Per-document, per-ingest-config persisted index: chunks.jsonl + parents.json + embeddings.npy. Brute-force exact cosine search
(thousands of chunks per filing: exact, reproducible, no vector DB)."""
import json
import os
import time
from pathlib import Path

import numpy as np

from ipo_rag.chunking import build_chunks
from ipo_rag.config import IngestConfig
from ipo_rag.corpus import pdf_path
from ipo_rag.parsing import extract_tables, load_pages

INDEX_ROOT = Path(__file__).resolve().parent.parent / "indexes"
EMBEDDERS = {  # name: (hf id, query prefix, passage prefix)
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-large": ("intfloat/e5-large-v2", "query: ", "passage: "),
}
_models = {}


def embedder(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(EMBEDDERS[name][0])
    return _models[name]


def embed(name, texts, kind, batch=32):
    _, qp, pp = EMBEDDERS[name]
    pre = qp if kind == "query" else pp
    return embedder(name).encode([pre + t for t in texts], batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def index_dir(cfg: IngestConfig, doc_id):
    suffix = "-sec" if os.environ.get("RAG_STRUCTURE", "sebi") == "sec" else ""   # section labels change the chunks, so they get their own index
    return INDEX_ROOT / (cfg.key() + suffix) / doc_id


def build_doc_index(doc, cfg: IngestConfig, max_pages=None, force=False, log=print):
    d = index_dir(cfg, doc["id"]) if not max_pages else index_dir(cfg, doc["id"] + f"__first{max_pages}")
    if (d / "meta.json").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    lock = d / ".building"
    if lock.exists() and time.time() - lock.stat().st_mtime < 5400:      # another process is building this index: wait for it
        while lock.exists() and not (d / "meta.json").exists() and time.time() - lock.stat().st_mtime < 5400:
            time.sleep(20)
        if (d / "meta.json").exists():
            return d
    lock.write_text(str(time.time()))
    try:
        return _build(doc, cfg, d, max_pages, log)
    finally:
        lock.unlink(missing_ok=True)


def _build(doc, cfg, d, max_pages, log):
    t0 = time.time()
    pages = load_pages(pdf_path(doc), max_pages)
    tables = extract_tables(pdf_path(doc), max_pages) if cfg.table_mode in ("flat", "markdown") else []
    t1 = time.time()
    chunks, parents = build_chunks(doc, pages, tables, cfg)
    emb = embed(cfg.embedder, [c["index_text"] for c in chunks], "passage")
    with open(d / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump(parents, open(d / "parents.json", "w"))
    np.save(d / "embeddings.npy", emb)
    json.dump({"doc": doc["id"], "sha256": doc["sha256"], "ingest": cfg.__dict__, "n_chunks": len(chunks), "n_parents": len(parents),
               "n_tables": sum(c["type"] == "table" for c in chunks), "parse_s": round(t1 - t0, 1), "total_s": round(time.time() - t0, 1)},
              open(d / "meta.json", "w"), indent=1)
    log(f"  built {d.name}: {len(chunks)} chunks ({sum(c['type']=='table' for c in chunks)} tables), {len(parents)} parents, {time.time()-t0:.0f}s")
    return d


def load_doc_index(d):
    chunks = [json.loads(l) for l in open(d / "chunks.jsonl")]
    return chunks, json.load(open(d / "parents.json")), np.load(d / "embeddings.npy")
