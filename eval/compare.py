"""Side-by-side report.  python eval/compare.py eval/results/*.judged.json   (or raw .json for no judge columns)"""
import json, os, sys
from statistics import mean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import mean_ci, percentile

runs = [json.load(open(p)) for p in sys.argv[1:] if not p.endswith("judgecache.json") and "judgecache" not in p]
W = 22


def fmt(xs, pct=False):
    m, lo, hi = mean_ci(xs)
    if m is None:
        return "-"
    f = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    ci = f" [{f(lo)}-{f(hi)}]" if lo is not None else ""
    return f"{f(m)}{ci} n={len([x for x in xs if x is not None])}"


def table(title, rows):
    print(f"\n## {title}")
    print(f"{'':24}" + "".join(f"{r['label'][:W-1]:{W}}" for r in runs))
    for name, fn in rows:
        print(f"{name:24}" + "".join(f"{str(fn(r)):{W}}" for r in runs))


def pr(v):
    return None if v is None else round(v, 1)


def gen(r):
    return r.get("generation", [])


def col(r, fn):
    return [fn(x) for x in gen(r) if not x["failed"]]


table("Run", [("model", lambda r: r["model"]), ("commit", lambda r: r["git"]), ("judge", lambda r: r.get("judge_model", "-")),
              ("questions / labelled", lambda r: f"{r['n_questions']} / {r['n_labelled']}"), ("ingest s", lambda r: r["ingest_s"])])
table("Ingestion", [("page coverage", lambda r: mean(d["page_coverage"] for d in r["ingestion"]["docs"])),
                    ("company name ok", lambda r: [d["company_ok"] for d in r["ingestion"]["docs"]]),
                    ("empty chunks", lambda r: r["ingestion"]["empty_chunks"]),
                    ("chunk chars p50", lambda r: r["ingestion"]["chunk_chars"]["p50"])])
cfgs = sorted({c for r in runs for c in r.get("retrieval", {})}, key=("dense", "bm25", "hybrid", "hybrid_rerank").index)
for m in ("hit@5", "recall@5", "precision@5", "ndcg@5", "mrr"):
    table(f"Retrieval {m} (ablation)", [(c, lambda r, c=c, m=m: r.get("retrieval", {}).get(c, {}).get("mean", {}).get(m, "-")) for c in cfgs])
if any(gen(r) for r in runs):
    table("Reliability & speed", [
        ("failure rate", lambda r: f"{sum(x['failed'] for x in gen(r))/len(gen(r)):.0%}"),
        ("gen latency p50 s", lambda r: pr(percentile([x["generation_s"] for x in gen(r) if not x["failed"]], 50))),
        ("gen latency p95 s", lambda r: pr(percentile([x["generation_s"] for x in gen(r) if not x["failed"]], 95)))])
    ans = lambda r: [x for x in gen(r) if x["answerable"] and not x["failed"]]
    table("Grounding (no LLM)", [
        ("has citation", lambda r: fmt([float(x["det"]["has_citation"]) for x in ans(r)], True)),
        ("citation accuracy", lambda r: fmt([x["det"]["citation_accuracy"] for x in ans(r)])),
        ("numeric support", lambda r: fmt([x["det"]["numeric_support"] for x in ans(r)])),
        ("key-fact recall", lambda r: fmt([x["det"]["fact_recall"] for x in ans(r)])),
        ("correct abstention", lambda r: fmt([float(x["det"]["correct_abstention"]) for x in gen(r) if not x["answerable"] and not x["failed"]], True)),
        ("false abstention", lambda r: fmt([float(x["det"]["abstained"]) for x in ans(r)], True))])
    table("RAG triad (LLM judge)", [(n, lambda r, k=k: fmt([x["scores"].get(k) for x in gen(r) if x.get("scores")]))
                                    for n, k in (("context relevance", "context_relevance"), ("faithfulness (claims)", "faithfulness"),
                                                 ("answer relevance", "relevance"), ("correctness vs ref", "correctness"))])
    cats = sorted({x["category"] for r in runs for x in gen(r)})
    table("Faithfulness by category", [(c, lambda r, c=c: fmt([x["scores"].get("faithfulness") for x in gen(r) if x.get("scores") and x["category"] == c])) for c in cats])
