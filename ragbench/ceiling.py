"""Ceiling test: give the generator the GOLD evidence pages as context (perfect retrieval) and score it like any other run.
The gap between this and the end-to-end result isolates retrieval loss from generation loss.
    python -m ragbench.ceiling --gen nano --split dev   (then: python -m ragbench.judge --run ceiling__nano__dev__generation)"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

from ipo_rag.config import BASELINE
from ipo_rag.generation import format_context, generate
from ragbench import gold, metrics as M
from ragbench.ablate import GENS
from ragbench.runner import RESULTS, aggregate, ctx_pages, run_dir

PAGES = {}


def page_text(doc, p):
    if doc not in PAGES:
        f = f"goldgen/out/pages/{doc}.jsonl"
        if os.path.exists(f):
            PAGES[doc] = {r["page"]: r["text"] for r in map(json.loads, open(f))}
        else:                                   # other corpora (e.g. financebench): read the PDF directly
            import fitz
            from ipo_rag.corpus import manifest, pdf_path
            PAGES[doc] = {i + 1: p.get_text() for i, p in enumerate(fitz.open(pdf_path(manifest()[doc])))}
    return PAGES[doc][p][:3500]


def gold_contexts(item):
    out = []
    for gi, grp in enumerate(item["evidence_groups"]):
        doc = item["docs"][gi] if item["type"] == "multi_document" else item["docs"][0]
        p = grp[0]
        out.append({"doc": doc, "text": page_text(doc, p), "page_start": p, "page_end": p, "chapter": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="nano", choices=list(GENS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    name = f"ceiling__{a.gen}"
    d = run_dir(name, a.split, "generation", False)
    if (d / "metrics.json").exists():
        print("exists", d.name)
        return
    cfg = replace(BASELINE, gen=replace(BASELINE.gen, model=GENS[a.gen], max_tokens=4096))
    items, gh = gold.load(a.split)
    items = [i for i in items if i["answerable"]]
    names = {k: k.replace("_", " ").title() for k in {x for i in items for x in i["docs"]}}
    def work(it):
        ctxs = gold_contexts(it)
        return ctxs, generate(cfg.gen, it["question"], format_context(ctxs, names))
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(work, items))
    rows = []
    for it, (ctxs, g) in zip(items, outs):
        texts = [c["text"] for c in ctxs]
        row = {"id": it["id"], "type": it["type"], "docs": it["docs"], "answerable": True, "question": it["question"], "answer": g["answer"], "failed": g["failed"],
               "error": g["error"], "contexts": texts, "context_pages": sorted(ctx_pages(ctxs)), "generation_s": g["seconds"], "retrieval_s": 0}
        if not g["failed"]:
            row["grounding"] = M.grounding_metrics(g["answer"], texts, ctx_pages(ctxs), it["key_facts"], True)
        rows.append(row)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = aggregate(rows)
    agg["failure_rate"] = sum(r["failed"] for r in rows) / len(rows)
    json.dump(agg, open(d / "metrics.json", "w"), indent=1)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"name": name, "config": asdict(cfg), "split": a.split, "mode": "generation", "oracle": True, "corpus": "gold-pages", "gold_sha256": gh, "git": git,
               "n_items": len(rows)}, open(d / "run.json", "w"), indent=1)
    print(d.name, "n =", len(rows), "failure rate", agg["failure_rate"])


if __name__ == "__main__":
    main()
