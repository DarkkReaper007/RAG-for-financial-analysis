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
from ipo_rag.keys import pool
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
_infer = threading.Lock()          # torch models (embedder, reranker) are loaded lazily and are not safe to initialise from two threads at once
_docs = {}


def pipe(hierarchy="parent"):
    """One pipeline per ingest config; indexes are pre-built by the experiments, so this only loads arrays."""
    with _lock:
        if hierarchy not in _pipes:
            cfg = with_changes(BASELINE, hierarchy=hierarchy, model=GEN_MODEL, max_tokens=4096)
            _pipes[hierarchy] = RAGPipeline(cfg, log=lambda *a: None).ingest()
        return _pipes[hierarchy]


def has_key():
    return bool(pool(BASELINE.gen.api_key_env).keys)


STATE = {"ready": False, "warm_s": None}


def warm():
    """Load both indexes, the embedder and the reranker once at start-up so the first question is not slow."""
    t = time.time()
    try:
        for h in ("parent", "flat"):
            p = pipe(h)
            with _infer:
                p.retrieve("Who is the registrar to the offer?", None)        # also builds the keyword index over all 12 filings (about 4 s)
                p.retrieve("Who is the registrar to the offer?", ["zomato"])
    except Exception as e:                                                # never block the server on warm-up
        print("warm-up skipped:", type(e).__name__, e, flush=True)
    STATE.update(ready=True, warm_s=round(time.time() - t, 1))
    print(f"ready: search models loaded in {STATE['warm_s']} s; answer writing is {'ON' if has_key() else 'OFF (no NVIDIA_API_KEYS)'}", flush=True)


def error_kind(err):
    e = (err or "").lower()
    if not e:
        return None
    if e == "no_key" or "no api key" in e:
        return "no_key"
    if "429" in e or "too many" in e or "rate" in e:
        return "rate_limit"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "401" in e or "403" in e:
        return "bad_key"
    if "empty_reply" in e:
        return "empty"
    return "other"


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


def nice(t):
    """Title-case a section name without the 'S artefact (Management'S -> Management's)."""
    return re.sub(r"'S\b", "'s", (t or "").title())


def chunk_view(c, rank=None):
    pe = c["page_end"]
    return {"rank": rank, "cid": c["id"], "doc": c["doc"], "company": man[c["doc"]]["company"].title(), "page_start": c["page_start"], "page_end": pe,
            "chapter": nice(c.get("chapter")), "type": c.get("type", "text"), "text": c["text"][:700], "score": c.get("rerank_score")}


def ctx_view(c):
    return {"doc": c["doc"], "company": man[c["doc"]]["company"].title(), "page_start": c["page_start"], "page_end": c["page_end"],
            "chapter": nice(c.get("chapter")), "expanded": c["expanded"], "chars": len(c["text"]), "text": c["text"][:1400]}


def citations(answer, ctxs=()):
    """Cited (filing, pages). The model often writes the placeholder [Company, Page: N] from the prompt, so when the name does not identify a filing
    it is taken from the retrieved passages that contain that page."""
    out = []
    for m in CITE.finditer(answer):
        name = m.group(1).strip().lower()
        doc = None if name in ("company", "the company") else next((d for d, v in man.items() if name in v["company"].lower() or v["company"].lower().split()[0] == name.split()[0]), None)
        pages = []
        for part in re.split(r"[,\s]+", m.group(2).strip()):
            a, _, b = part.replace("–", "-").partition("-")
            if a.isdigit():
                pages += list(range(int(a), int(b or a) + 1)) if b.isdigit() and int(b) - int(a) < 6 else [int(a)]
        if doc is None and pages:
            doc = next((c["doc"] for c in ctxs if any(c["page_start"] <= p <= c["page_end"] for p in pages)), None)
        out.append({"label": m.group(0), "doc": doc, "pages": sorted(set(pages))})
    return out


def run(req: Ask, saved=None):
    """saved: a stored answer to replay; search is always recomputed (it is deterministic and fast), so the trace is always current."""
    p = pipe(req.hierarchy)
    docs = req.docs or None
    cfg_q = replace(p.cfg.query, retrieval_mode=req.mode, verify=req.verify)
    p2 = RAGPipeline(replace(p.cfg, query=cfg_q), log=lambda *a: None)
    p2.corpus, p2.doc_ids = p.corpus, p.doc_ids
    corpus = _subset(p.corpus, docs) if docs else p.corpus
    wall = time.time()
    with _infer:
        r = p2.retrieve(req.question, docs)
        n = cfg_q.top_k * cfg_q.candidate_depth
        stages = {}
        for m in ("dense", "bm25"):                                      # what each retriever alone would return (no rerank), for the trace
            cand, _ = search(corpus, req.question, replace(cfg_q, retrieval_mode=m, rerank=False))
            stages[m] = [chunk_view(c, i + 1) for i, c in enumerate(cand[:n])]
    stages["fused"] = [chunk_view(c, i + 1) for i, c in enumerate(r["candidates"])]
    stages["reranked"] = [chunk_view(c, i + 1) for i, c in enumerate(r["final"])]
    rank = lambda lst: {c["cid"]: c["rank"] for c in lst}
    d_r, b_r, f_r = rank(stages["dense"]), rank(stages["bm25"]), rank(stages["fused"])
    for c in stages["reranked"]:                                         # where each passage that was used came from
        c["from_dense"], c["from_bm25"], c["from_fused"] = d_r.get(c["cid"]), b_r.get(c["cid"]), f_r.get(c["cid"])
    if saved:
        res = {"answer": saved["answer"], "failed": False, "gen": {"error": None, "seconds": (saved.get("timing") or {}).get("generation_s")},
               "verification_action": saved.get("verification_action")}
    elif has_key():
        res = p2.answer_from(req.question, r)
        spent = res["gen"].get("seconds") or 0
        for _ in range(2):                                               # the reasoning model sometimes returns an empty reply or a transient error: ask again (up to twice)
            if not res["failed"] or error_kind(res["gen"].get("error")) not in ("empty", "other"):
                break
            res = p2.answer_from(req.question, r)
            spent += res["gen"].get("seconds") or 0
            res["gen"]["seconds"] = round(spent, 2)
    else:                                                              # no generator key on this server: still return the search
        res = {"answer": "", "failed": True, "gen": {"error": "no_key", "seconds": None}, "verification_action": None}
    ans = res["answer"]
    pages = {(c["doc"], q) for c in r["contexts"] for q in range(c["page_start"], c["page_end"] + 1)}
    chk = check(ans, [c["text"] for c in r["contexts"]], {q for _, q in pages}) if ans and not res["failed"] else None
    gen_s = res["gen"].get("seconds")
    err = res["gen"].get("error") if res["failed"] else None
    return {"question": req.question, "answer": ans, "failed": res["failed"], "error": err, "error_kind": error_kind(err),
            "abstained": ans.startswith(ABSTAIN), "citations": citations(ans, r["contexts"]),
            "audit": chk and {"ok": chk["ok"], "numbers_checked": len(numbers(ans)), "unsupported": chk["unsupported_numbers"], "bad_citations": chk["invalid_citations"]},
            "verification_action": res.get("verification_action"), "stages": stages, "contexts": [ctx_view(c) for c in r["contexts"]],
            "corpus": {"chunks": len(corpus.chunks), "filings": len({c["doc"] for c in corpus.chunks}), "per_retriever": n},
            "timing": {"retrieval_s": r["retrieval_s"], "generation_s": gen_s, "total_s": round(r["retrieval_s"] + (gen_s or 0), 2),
                       "wall_s": round(time.time() - wall, 2)},
            "settings": {"mode": req.mode, "verify": req.verify, "hierarchy": req.hierarchy, "docs": req.docs}}


def key(req: Ask):
    return hashlib.sha1(json.dumps([req.question.strip(), sorted(req.docs), req.mode, req.verify, req.hierarchy]).encode()).hexdigest()[:16]


def ask_cached(req: Ask):
    f = CACHE / f"{key(req)}.json"
    saved = json.load(open(f)) if f.exists() and not req.refresh else None
    if saved:
        return {**run(req, saved), "cached": True}
    out = run(req)
    if out["failed"] and f.exists():                                      # API down: replay the stored answer
        return {**run(req, json.load(open(f))), "cached": True}
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


@app.get("/api/status")
def api_status():
    return {"answers_live": has_key(), "ready": STATE["ready"], "warm_s": STATE["warm_s"], "model": GEN_MODEL.split("/")[-1],
            "saved_answers": len(list(CACHE.glob("*.json")))}


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


@app.on_event("startup")
def _startup():
    threading.Thread(target=warm, daemon=True).start()


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
