"""python -m ipo_rag.cli ingest --docs zomato [--set chunker=fixed hierarchy=flat]
   python -m ipo_rag.cli query "What is the price band?" --docs zomato [--no-llm]"""
import argparse
import json

from ipo_rag.config import BASELINE, with_changes
from ipo_rag.pipeline import RAGPipeline


def coerce(v):
    return {"true": True, "false": False}.get(v.lower(), int(v) if v.isdigit() else v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "query"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. chunker=fixed retrieval_mode=dense")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    cfg = with_changes(BASELINE, **{k: coerce(v) for k, v in (s.split("=", 1) for s in a.set)})
    p = RAGPipeline(cfg, a.docs, a.max_pages).ingest()
    if a.cmd == "query":
        r = p.retrieve(a.question) if a.no_llm else p.answer(a.question)
        for c in r["contexts"]:
            print(f"--- p.{c['page_start']}-{c['page_end']} {c['chapter']} expanded={c['expanded']}\n{c['text'][:300]}")
        if not a.no_llm:
            print("\nANSWER:", r["answer"], "\n", json.dumps({k: r[k] for k in ("verification", "verification_action")}), r["gen"]["seconds"], "s")


if __name__ == "__main__":
    main()
