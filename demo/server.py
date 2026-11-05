"""Prospectus Desk: local demo backend for the IPO RAG pipeline.
  source .venv/bin/activate; export NVIDIA_API_KEYS=...   # only needed for questions that are not in demo/cache
  python -m demo.server            # http://127.0.0.1:8000
  python -m demo.server --warm     # pre-compute the example questions into demo/cache (replay works without an API key)
Answers are cached per (question, filings, settings); a cached answer is flagged and replayed when the API is unreachable."""
import hashlib
import json
import re
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ipo_rag.config import BASELINE, with_changes
from ipo_rag.corpus import manifest, pdf_path
from ipo_rag.pipeline import ABSTAIN, RAGPipeline, _subset
from ipo_rag.retrieval import search
from ipo_rag.verification import check, numbers

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
GOLD = ROOT.parent / "eval" / "gold" / "gold_v1.json"
GEN_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"       # generator chosen by the pre-registered rule
CITE = re.compile(r"\[([^\[\],]+?),\s*Page[s]?:?\s*([\d\s,\-–]+)\]", re.I)

app = FastAPI(title="Prospectus Desk")
man = manifest()
_pipes, _lock = {}, threading.Lock()
_docs = {}


def pipe(hierarchy="parent"):
    """One pipeline per ingest config; indexes are pre-built by the experiments, so this only loads arrays."""
    with _lock:
        if hierarchy not in _pipes:
            cfg = with_changes(BASELINE, hierarchy=hierarchy, model=GEN_MODEL, max_tokens=4096)
            _pipes[hierarchy] = RAGPipeline(cfg, log=lambda *a: None).ingest()
        return _pipes[hierarchy]


def pdf(doc):
    if doc not in _docs:
        _docs[doc] = fitz.open(pdf_path(man[doc]))
    return _docs[doc]


class Ask(BaseModel):
    question: str
    docs: list[str] = []
    mode: str = "hybrid"          # dense | bm25 | hybrid
    verify: str = "off"           # off | on
    hierarchy: str = "parent"     # parent | flat
    refresh: bool = False


def chunk_view(c, rank=None):
    pe = c["page_end"]
    return {"rank": rank, "doc": c["doc"], "company": man[c["doc"]]["company"].title(), "page_start": c["page_start"], "page_end": pe,
            "chapter": (c.get("chapter") or "").title(), "type": c.get("type", "text"), "text": c["text"][:700], "score": c.get("rerank_score")}


def ctx_view(c):
    return {"doc": c["doc"], "company": man[c["doc"]]["company"].title(), "page_start": c["page_start"], "page_end": c["page_end"],
            "chapter": (c.get("chapter") or "").title(), "expanded": c["expanded"], "chars": len(c["text"]), "text": c["text"][:1400]}


def citations(answer):
    out = []
    for m in CITE.finditer(answer):
        name = m.group(1).strip().lower()
        doc = next((d for d, v in man.items() if name in v["company"].lower() or v["company"].lower().split()[0] == name.split()[0]), None)
        pages = []
        for part in re.split(r"[,\s]+", m.group(2).strip()):
            a, _, b = part.replace("–", "-").partition("-")
            if a.isdigit():
                pages += list(range(int(a), int(b or a) + 1)) if b.isdigit() and int(b) - int(a) < 6 else [int(a)]
        out.append({"label": m.group(0), "doc": doc, "pages": sorted(set(pages))})
    return out


def run(req: Ask):
    p = pipe(req.hierarchy)
    docs = req.docs or None
    cfg_q = replace(p.cfg.query, retrieval_mode=req.mode, verify=req.verify)
    p2 = RAGPipeline(replace(p.cfg, query=cfg_q), log=lambda *a: None)
    p2.corpus, p2.doc_ids = p.corpus, p.doc_ids
    corpus = _subset(p.corpus, docs) if docs else p.corpus
    t = time.time()
    r = p2.retrieve(req.question, docs)
    stages = {}
    for m in ("dense", "bm25"):                                          # what each retriever alone would return (no rerank), for the trace
        cand, _ = search(corpus, req.question, replace(cfg_q, retrieval_mode=m, rerank=False))
        stages[m] = [chunk_view(c, i + 1) for i, c in enumerate(cand[:5])]
    stages["fused"] = [chunk_view(c, i + 1) for i, c in enumerate(r["candidates"])]
    stages["reranked"] = [chunk_view(c, i + 1) for i, c in enumerate(r["final"])]
    res = p2.answer_from(req.question, r)
    ans = res["answer"]
    pages = {(c["doc"], q) for c in r["contexts"] for q in range(c["page_start"], c["page_end"] + 1)}
    chk = check(ans, [c["text"] for c in r["contexts"]], {q for _, q in pages}) if ans and not res["failed"] else None
    return {"question": req.question, "answer": ans, "failed": res["failed"], "error": res["gen"].get("error"),
            "abstained": ans.startswith(ABSTAIN), "citations": citations(ans),
            "audit": chk and {"ok": chk["ok"], "numbers_checked": len(numbers(ans)), "unsupported": chk["unsupported_numbers"], "bad_citations": chk["invalid_citations"]},
            "verification_action": res.get("verification_action"), "stages": stages, "contexts": [ctx_view(c) for c in r["contexts"]],
            "timing": {"retrieval_s": r["retrieval_s"], "generation_s": res["gen"].get("seconds"), "total_s": round(time.time() - t, 2)},
            "settings": {"mode": req.mode, "verify": req.verify, "hierarchy": req.hierarchy, "docs": req.docs}}


def key(req: Ask):
    return hashlib.sha1(json.dumps([req.question.strip(), sorted(req.docs), req.mode, req.verify, req.hierarchy]).encode()).hexdigest()[:16]


def ask_cached(req: Ask):
    f = CACHE / f"{key(req)}.json"
    if f.exists() and not req.refresh:
        return {**json.load(open(f)), "cached": True}
    out = run(req)
    if out["failed"] and f.exists():                                      # API down: replay the stored answer
        return {**json.load(open(f)), "cached": True}
    if not out["failed"]:
        json.dump(out, open(f, "w"), ensure_ascii=False)
    return {**out, "cached": False}


@app.post("/api/ask")
def api_ask(req: Ask):
    if not req.question.strip():
        raise HTTPException(400, "Type a question first.")
    try:
        return ask_cached(req)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/docs")
def api_docs():
    return [{"id": d, "company": v["company"].title(), "type": v["doc_type"].upper(), "pages": v.get("pages")} for d, v in man.items()]


def examples():
    g = json.load(open(GOLD))
    items = g["items"] if isinstance(g, dict) else g
    want = ["t-fac-06", "t-tab-05", "t-lis-11", "t-mul-02", "t-una-03", "t-num-09"]
    by = {i["id"]: i for i in items}
    label = {"factoid": "Fact", "table": "Table", "list": "List", "multi_document": "Two filings", "unanswerable": "Not in the document", "numeric": "Figure"}
    return [{"id": i, "label": label[by[i]["type"]], "question": by[i]["question"], "docs": by[i]["docs"], "reference": by[i]["reference_answer"],
             "gold_pages": sorted({p for g in by[i]["evidence_groups"] for p in g}), "answerable": by[i]["answerable"]} for i in want if i in by]


@app.get("/api/examples")
def api_examples():
    return examples()


@app.get("/api/results")
def api_results():
    return json.load(open(ROOT / "results.json"))


@app.get("/api/page/{doc}/{page}")
def api_page(doc: str, page: int, hl: str = "", zoom: float = 1.7):
    if doc not in man or not 1 <= page <= len(pdf(doc)):
        raise HTTPException(404, "No such page")
    pg = pdf(doc)[page - 1]
    for term in {t for t in hl.split("|") if len(t) >= 2}:
        for rect in pg.search_for(term):
            pg.add_highlight_annot(rect)
    png = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    if "--warm" in sys.argv:
        for ex in examples():
            for kw in ({}, {"mode": "dense"}, {"verify": "on"}, {"hierarchy": "flat"}):
                t = time.time()
                o = ask_cached(Ask(question=ex["question"], docs=ex["docs"], **kw))
                print(ex["id"], kw, "cached" if o["cached"] else "computed", "failed" if o["failed"] else "ok", round(time.time() - t, 1), flush=True)
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
