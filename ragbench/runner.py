"""Run one configuration over the gold set and save per-question results + aggregates.

  python -m ragbench.runner --name baseline --split dev --mode retrieval
  python -m ragbench.runner --name verify_on --split dev --mode generation --set verify=on
Retrieval is scored over the OPEN corpus (all filings together, no router) unless --oracle restricts each question to its own
filing(s) (an upper bound). Results: ragbench/results/<name>__<split>__<mode>[__oracle]/{run.json,rows.jsonl,metrics.json}
"""
import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ipo_rag.config import BASELINE, with_changes
from ipo_rag.pipeline import RAGPipeline
from ragbench import gold, metrics as M
from ragbench.stats import mean_ci

RESULTS = Path(__file__).resolve().parent / "results"
CAND_KS, FINAL_KS = (1, 3, 5, 10), (1, 3, 5)


def coerce(v):
    return {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)


def run_dir(name, split, mode, oracle):
    return RESULTS / f"{name}__{split}__{mode}{'__oracle' if oracle else ''}"


def ctx_pages(ctxs):
    return {p for c in ctxs for p in range(c["page_start"], c["page_end"] + 1)}


def score_retrieval(item, r):
    g = gold.groups_of(item)
    row = {"cand": M.retrieval_metrics(r["candidates"], g, CAND_KS), "final": M.retrieval_metrics(r["final"], g, FINAL_KS),
           "ctx": {k: v for k, v in M.retrieval_metrics(r["contexts"], g, (10,)).items() if k.endswith("@10")}}
    row["retrieved"] = [f"{c['doc']}:{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "") for c in r["candidates"]]
    return row


def flat(row):
    """{'cand.recall@5': v, ...} for aggregation."""
    return {f"{sec}.{k}": v for sec in ("cand", "final", "ctx", "grounding", "judge") for k, v in (row.get(sec) or {}).items()}


def aggregate(rows):
    def agg(rs):
        keys = sorted({k for r in rs for k in flat(r)})
        out = {}
        for k in keys:
            m, lo, hi = mean_ci([flat(r).get(k) for r in rs])
            if m is not None:
                out[k] = {"mean": round(m, 4), "lo": None if lo is None else round(lo, 4), "hi": None if hi is None else round(hi, 4)}
        return {"n": len(rs), "metrics": out}
    types = sorted({r["type"] for r in rows})
    return {"overall": agg(rows), "by_type": {t: agg([r for r in rows if r["type"] == t]) for t in types}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--mode", default="retrieval", choices=["retrieval", "generation"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = run_dir(a.name, a.split, a.mode, a.oracle)
    if (d / "metrics.json").exists() and not a.force:
        print("exists, skipping:", d.name)
        return
    cfg = with_changes(BASELINE, **{k: coerce(v) for k, v in (s.split("=", 1) for s in a.set)})
    items, gh = gold.load(a.split)
    if a.mode == "retrieval":
        items = [i for i in items if i["answerable"]]
    if a.limit:
        items = items[:a.limit]
    t0 = time.time()
    pipe = RAGPipeline(cfg).ingest()
    build_s = round(time.time() - t0, 1)
    rows = []
    for it in items:
        docs = it["docs"] if a.oracle else None
        row = {"id": it["id"], "type": it["type"], "docs": it["docs"], "answerable": it["answerable"]}
        if a.mode == "retrieval":
            r = pipe.retrieve(it["question"], docs)
            row.update(score_retrieval(it, r) if it["answerable"] else {})
            row["retrieval_s"] = r["retrieval_s"]
        else:
            r = pipe.answer(it["question"], docs)
            if it["answerable"]:
                row.update(score_retrieval(it, r))
            ctexts = [c["text"] for c in r["contexts"]]
            row.update({"question": it["question"], "answer": r["answer"], "failed": r["failed"], "error": r["gen"]["error"], "contexts": ctexts,
                        "context_pages": sorted(ctx_pages(r["contexts"])), "retrieval_s": r["retrieval_s"], "generation_s": r["gen"]["seconds"],
                        "verification": r["verification"], "verification_action": r["verification_action"]})
            if not r["failed"]:
                row["grounding"] = M.grounding_metrics(r["answer"], ctexts, ctx_pages(r["contexts"]), it["key_facts"], it["answerable"])
        rows.append(row)
        print(f"  {row['id']} {a.mode}", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = aggregate(rows)
    if a.mode == "generation":
        agg["failure_rate"] = sum(r["failed"] for r in rows) / len(rows)
    json.dump(agg, open(d / "metrics.json", "w"), indent=1)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"name": a.name, "config": asdict(cfg), "split": a.split, "mode": a.mode, "oracle": a.oracle, "gold_sha256": gh, "git": git,
               "n_items": len(rows), "build_s": build_s, "run_s": round(time.time() - t0, 1)}, open(d / "run.json", "w"), indent=1)
    m = agg["overall"]["metrics"]
    print(f"\n{d.name}: n={agg['overall']['n']}  " + "  ".join(f"{k}={m[k]['mean']:.3f}" for k in ("cand.hit@5", "cand.recall@5", "cand.ndcg@5", "final.recall@5", "final.mrr") if k in m))


if __name__ == "__main__":
    main()
