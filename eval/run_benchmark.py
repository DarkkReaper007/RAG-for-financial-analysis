"""Run the pipeline over the gold set and save RAW results (no LLM judging here).

    python eval/run_benchmark.py --label C_mistral --model mistral:7b --mode all
Modes: ingestion (parse/chunk sanity) | retrieval (ablation, no LLM) | generation | all
Run from the repo root of the code under test.
"""
import argparse, inspect, json, os, sys, time
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

FAIL = ("Error generating response", "Error: Could not connect", "[Error")
CONFIGS = ("dense", "bm25", "hybrid", "hybrid_rerank")


def build_pipeline(model, url, timeout):
    params = inspect.signature(RAGPipeline.__init__).parameters
    kw = dict(use_qdrant=False, use_reranker=True)
    if "model_config" in params:
        kw["model_config"] = SimpleNamespace(model=model, url=url, temperature=0.1, max_tokens=2048, timeout=timeout)
    else:
        kw.update(ollama_model=model, ollama_url=url)
    return RAGPipeline(**kw)


def ingestion_report(pipe, stats, gold_docs):
    chunks = pipe.all_chunks
    retr = [c for c in chunks if c.chunk_type in ("child", "table")]
    lens = [len(c.text) for c in retr]
    rep = {"docs": []}
    for d in stats["documents"]:
        pages = {int(c.metadata.get("page_start", 0)) for c in retr if c.metadata.get("filename") == d["filename"]}
        exp = (gold_docs.get(d["filename"]) or {}).get("company")
        rep["docs"].append({
            "filename": d["filename"], "company": d["company_name"], "company_ok": (d["company_name"] == exp) if exp else None,
            "pages": d["total_pages"], "sections": d["sections"], "tables": d["tables"], "chunks": d["chunks"],
            "page_coverage": round(len(pages) / d["total_pages"], 3), "parse_chunk_s": d["timing"]["total_seconds"]})
    rep["retrievable_chunks"] = len(retr)
    rep["empty_chunks"] = sum(1 for n in lens if n < 20)
    rep["chunk_chars"] = {"p5": M.percentile(lens, 5), "p50": M.percentile(lens, 50), "p95": M.percentile(lens, 95)}
    return rep


def ranked(pipe, cfg, q, k, company):
    """Return chunk dicts in true rank order for a retrieval config."""
    r = pipe.retriever
    if cfg == "hybrid_rerank":
        chunks = r.retrieve(q, top_k=k, company_filter=company, expand_parents=False)["chunks"]
        return sorted(chunks, key=lambda c: c.get("rerank_score", c.get("rrf_score", 0)), reverse=True)
    dense = pipe.vector_store.search(q, top_k=k * 2, company_filter=company)
    sparse = pipe.bm25_index.search(q, top_k=k * 2, company_filter=company)
    if cfg == "dense":
        return dense[:k]
    if cfg == "bm25":
        return sparse[:k]
    return r._reciprocal_rank_fusion(dense, sparse)[:k]


def run_retrieval(pipe, gold, depth, tol):
    out = {}
    for cfg in CONFIGS:
        rows = []
        for q in gold:
            if not q["answerable"] or not q["relevant_pages"]:
                continue
            t = time.time()
            ch = ranked(pipe, cfg, q["question"], depth, q.get("company"))
            pages = [int(c.get("page_start", 0)) for c in ch]
            rows.append({"id": q["id"], "category": q["category"], "pages": pages,
                         "latency_s": round(time.time() - t, 3), **M.retrieval_scores(pages, q["relevant_pages"], tol=tol)})
        keys = [k for k in (rows[0] if rows else {}) if k not in ("id", "category", "pages", "latency_s")]
        out[cfg] = {"n": len(rows), "mean": {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys} if rows else {}, "rows": rows}
        print(f"  retrieval[{cfg}] n={len(rows)}", out[cfg]["mean"].get("hit@5"), flush=True)
    return out


def ctx_text(c):
    pt = c.get("parent_text")
    return pt if pt and len(pt) < 3000 else c.get("text", "")  # exactly what the generator sees


def run_generation(pipe, gold, top_k, tol):
    rows = []
    for q in gold:
        t = time.time()
        rr = pipe.retriever.retrieve(query=q["question"], top_k=top_k, company_filter=q.get("company"))
        ret_s = time.time() - t
        chunks = rr["chunks"]
        g = pipe.generator.generate(q["question"], chunks)
        ans = g.get("answer", "")
        ctxs = [ctx_text(c) for c in chunks]
        pages = [int(c.get("page_start", 0)) for c in chunks]
        failed = ans.startswith(FAIL)
        row = {"id": q["id"], "category": q["category"], "question": q["question"], "answerable": q["answerable"],
               "answer": ans, "failed": failed, "contexts": ctxs, "pages": pages,
               "retrieval_s": round(ret_s, 2), "generation_s": round(g.get("timing", {}).get("total_seconds", 0), 2),
               "eval_tokens": g.get("timing", {}).get("eval_count"),
               "det": {} if failed else {
                   "has_citation": M.has_citation(ans), "citation_accuracy": M.citation_accuracy(ans, pages),
                   "numeric_support": M.numeric_support(ans, ctxs), "abstained": M.is_abstention(ans),
                   "fact_recall": M.fact_recall(ans, q.get("key_facts"))}}
        if q["answerable"] and q["relevant_pages"]:
            row["retrieval"] = M.retrieval_scores(pages, q["relevant_pages"], ks=(3, 5), tol=tol)
        if not failed and not q["answerable"]:
            row["det"]["correct_abstention"] = row["det"]["abstained"]
        rows.append(row)
        print(f"  gen {q['id']} failed={failed} {row['generation_s']}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--depth", type=int, default=10, help="retrieval ablation depth")
    ap.add_argument("--tol", type=int, default=0, help="page tolerance for relevance")
    ap.add_argument("--mode", choices=("ingestion", "retrieval", "generation", "all"), default="all")
    ap.add_argument("--gold", default="eval/gold_questions.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    G = json.load(open(a.gold))
    gold = G["questions"][: a.limit] if a.limit else G["questions"]
    docs = sorted({q["doc"] for q in gold})
    pipe = build_pipeline(a.model, a.url, a.timeout)
    t0 = time.time()
    stats = pipe.ingest(pdf_paths=[f"data/prospectuses/{d}" for d in docs], recreate_index=True)
    res = {"label": a.label, "model": a.model, "top_k": a.top_k, "tol": a.tol, "ingest_s": round(time.time() - t0, 1),
           "git": os.popen("git rev-parse --short HEAD").read().strip(), "n_questions": len(gold),
           "n_labelled": sum(1 for q in gold if q["answerable"] and q["relevant_pages"])}
    res["ingestion"] = ingestion_report(pipe, stats, G.get("docs", {}))
    if a.mode in ("retrieval", "all"):
        res["retrieval"] = run_retrieval(pipe, gold, a.depth, a.tol)
    if a.mode in ("generation", "all"):
        res["generation"] = run_generation(pipe, gold, a.top_k, a.tol)
    out = a.out or f"eval/results/{a.label}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
