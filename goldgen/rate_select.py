"""Stage 3b/3c: LLM rating (naturalness, investor value) of passing candidates, then quota-based selection.
    python goldgen/rate_select.py   ->  goldgen/out/selected.jsonl"""
import argparse, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from goldgen.common import OUT, ROOT, chat_fb, manifest, parse_json, read_jsonl, write_jsonl

RATE = """You are reviewing a question written for an evaluation set about Indian IPO offer documents.
QUESTION: {q}
ANSWER: {a}

Rate two things from 1 (poor) to 5 (excellent):
- natural: does it read like a real, well-formed question a person would ask (not awkward, not a fill-in-the-blank)?
- value: would a retail investor or analyst plausibly care about this fact (business, financials, offer terms, risks, governance, use of proceeds, litigation)? Trivial administrative detail scores 1-2.
Reply with JSON only: {{"natural": n, "value": n}}"""
QUOTA = {"factoid": 2, "numeric": 2, "table": 2, "list": 2}     # per document; definitions: up to 1 per sampled doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--fallback", default="groq:openai/gpt-oss-120b", help="model for ratings that are not cached")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rows = [r for r in read_jsonl(OUT / "filtered.jsonl") if r["status"] == "pass"]
    excl = json.load(open(ROOT / "goldgen" / "review_exclusions.json")) if (ROOT / "goldgen" / "review_exclusions.json").exists() else {}

    def rate(r):
        txt, used = chat_fb(a.verifier, a.fallback, RATE.format(q=r["question"], a=r["answer"]), 0, 3000, 1500)
        j = parse_json(txt) or {}
        r["natural"], r["value"], r["rated_by"] = j.get("natural"), j.get("value"), used
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(rate, rows))
    rng = random.Random(11)
    by = defaultdict(list)
    for r in rows:
        if r["cid"] not in excl:                       # reviewer exclusions
            by[(r["doc"], r["target_type"])].append(r)
    sel, short = [], []
    for did in manifest():
        for t, n in {**QUOTA, "definition": 1}.items():
            pool = by.get((did, t), [])
            def score(r): return (r.get("natural") or 0) + (r.get("value") or 0) + rng.random() * 0.1
            good = sorted([r for r in pool if (r.get("natural") or 0) >= 4 and (r.get("value") or 0) >= 3], key=score, reverse=True)
            pool = [r for r in pool if t == "definition" or (r.get("value") or 0) >= 2]    # drop value<=1 trivia unless rating missing
            rest = sorted([r for r in pool if r not in good], key=score, reverse=True)
            take = (good + rest)[:n]
            sel += take
            if len(take) < n and t != "definition":
                short.append((did, t, n - len(take)))
    write_jsonl(OUT / "selected.jsonl", sel)
    print("passing:", len(rows), "selected:", len(sel), dict(Counter(r["target_type"] for r in sel)))
    print("rating means: natural=%.2f value=%.2f" % (sum(r.get("natural") or 0 for r in sel) / len(sel), sum(r.get("value") or 0 for r in sel) / len(sel)))
    print("quota shortfalls (doc, type, missing):", short)


if __name__ == "__main__":
    main()
