"""Stage 4: find other pages of the same filing that also answer a selected question (filings repeat facts).
Candidates: exact quote match, pages containing all answer numbers, BM25 top-5; each confirmed by the verifier model.
    python goldgen/expand_evidence.py  ->  goldgen/out/evidence.jsonl"""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import OUT, chat_fb, norm, read_jsonl, write_jsonl
from ipo_rag.retrieval import tokenize
from ipo_rag.verification import numbers

JUDGE = """PAGE:
\"\"\"
{page}
\"\"\"

QUESTION: {q}
REFERENCE ANSWER: {a}

Does this PAGE explicitly state the information needed to give the same answer to the QUESTION? Reply with exactly one word: yes or no."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--fallback", default="groq:openai/gpt-oss-120b", help="judge for pages whose NVIDIA verdict is not cached")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-cands", type=int, default=6)
    a = ap.parse_args()
    items = read_jsonl(OUT / "selected.jsonl")
    docs = {}
    for d in {r["doc"] for r in items}:
        pg = read_jsonl(OUT / "pages" / f"{d}.jsonl")
        docs[d] = {"pages": pg, "norm": {p["page"]: norm(p["text"]) for p in pg}, "nums": {p["page"]: numbers(p["text"], False) for p in pg},
                   "bm25": BM25Okapi([tokenize(p["text"]) for p in pg])}

    def work(r):
        D, src = docs[r["doc"]], r["page"]
        qn, nums = norm(r["quote"]), numbers(str(r["answer"]), False)
        cands = {p for p, t in D["norm"].items() if p != src and qn in t}
        if nums:
            cands |= {p for p, s in D["nums"].items() if p != src and nums <= s}
        top = sorted(range(len(D["pages"])), key=lambda i: -D["bm25"].get_scores(tokenize(r["question"] + " " + str(r["answer"])))[i])[:5] if False else None
        sc = D["bm25"].get_scores(tokenize(r["question"] + " " + str(r["answer"])))
        cands |= {D["pages"][i]["page"] for i in sc.argsort()[::-1][:5] if D["pages"][i]["page"] != src}
        cands = sorted(cands)[: a.max_cands]
        keep, unchecked, judged = [], [], {}
        for p in cands:
            out, used = chat_fb(a.verifier, a.fallback, JUDGE.format(page=next(x for x in D["pages"] if x["page"] == p)["text"][:7000], q=r["question"], a=r["answer"]), 0, 4000, 800)
            judged[p] = used
            if not out.strip():
                unchecked.append(p)            # API gave no reply (rate limit/timeout): NOT the same as "no"
            elif out.strip().lower().startswith("yes"):
                keep.append(p)
        return {**r, "evidence_groups": [sorted({src, *keep})], "evidence_checked": cands, "evidence_unchecked": unchecked, "evidence_judged_by": judged}

    rows = items
    for attempt in range(4):                  # re-run items that still have unchecked candidates (cache makes finished checks free)
        todo = [i for i, r in enumerate(rows) if attempt == 0 or r.get("evidence_unchecked")]
        if not todo:
            break
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(work, [items[i] for i in todo]))
        rows = list(rows) if attempt else [None] * len(items)
        for i, r in zip(todo, res):
            rows[i] = r
        print(f"pass {attempt}: unchecked items remaining = {sum(bool(r.get('evidence_unchecked')) for r in rows)}", flush=True)
    write_jsonl(OUT / "evidence.jsonl", rows)
    left = sum(len(r.get("evidence_unchecked", [])) for r in rows)
    if left:
        print(f"WARNING: {left} candidate pages could not be checked; their evidence is incomplete")
    ext = [len(r["evidence_groups"][0]) - 1 for r in rows]
    print(f"{len(rows)} items; with extra evidence pages: {sum(e > 0 for e in ext)}; mean extra pages {sum(ext)/len(ext):.2f}; max {max(ext)}")


if __name__ == "__main__":
    main()
