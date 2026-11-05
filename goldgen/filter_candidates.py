"""Stage 3: automatic filters on candidates. Writes goldgen/out/filtered.jsonl with status + reasons + tags.
    python goldgen/filter_candidates.py [--verifier nvidia/nemotron-3.5-lightning-30b-a3b]"""
import argparse, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import ALIASES, OUT, chat, norm, read_jsonl, write_jsonl

BAD_Q = re.compile(r"\b(this page|the page|the table|the above|above|below|the following table|this table|the document|this document|the figure)\b", re.I)
STOP = set("the a an of in on at to for and or is are was were be by with from as that this which what who how many much does do did its it their".split())


def grams(text, n):
    t = re.findall(r"[a-z0-9₹\.]+", norm(text))
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def content(text):
    return {w for w in re.findall(r"[a-z0-9]+", norm(text)) if w not in STOP}


VERIFY = """PAGE:
\"\"\"
{page}
\"\"\"

QUESTION: {q}
CANDIDATE ANSWER: {a}

Can the QUESTION be answered from the PAGE alone, and is the CANDIDATE ANSWER correct and complete according to the PAGE?
Reply with exactly one word: yes or no."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()
    pages = {}
    rows = read_jsonl(OUT / "candidates.jsonl")
    for d in {r["doc"] for r in rows}:
        pages[d] = {p["page"]: p["text"] for p in read_jsonl(OUT / "pages" / f"{d}.jsonl")}

    seen = []
    for r in rows:
        why = []
        if r["skip"] or not (r["question"] and r["answer"] and r["quote"]):
            r["status"], r["reasons"] = "skipped", [r.get("skip_reason") or "no_output"]
            continue
        page = pages[r["doc"]][r["page"]]
        q, ans, quote = r["question"], str(r["answer"]), r["quote"]
        if norm(quote) not in norm(page):
            why.append("quote_not_on_page")
        if "●" in ans or "[•]" in ans or len(ans.strip()) < 1:
            why.append("placeholder_or_empty_answer")
        if len(q.split()) < 8:
            why.append("question_too_short")
        if re.search(r"\bfor (the )?" + re.escape(ALIASES[r["doc"]][0]) + r"\s*\??$", q, re.I):
            why.append("appended_company_name")
        if BAD_Q.search(q):
            why.append("not_self_contained")
        if not any(al.lower() in q.lower() for al in ALIASES[r["doc"]]):
            why.append("company_not_named")
        if grams(q, 6) & grams(quote, 6):
            why.append("copies_6gram_from_quote")
        qs = content(q)
        if any(len(qs & content(s)) / max(1, len(qs | content(s))) > 0.8 for s in seen):
            why.append("near_duplicate")
        r["reasons"] = why
        r["tags"] = {"lexical_overlap": round(len(qs & content(quote)) / max(1, len(qs)), 2)}
        seen.append(q)

    todo = [r for r in rows if "reasons" in r and not r["reasons"] and "status" not in r]

    def verify(r):
        out = chat(a.verifier, VERIFY.format(page=pages[r["doc"]][r["page"]][:7000], q=r["question"], a=r["answer"]), temperature=0, max_tokens=4000)
        r["verifier_reply"] = out
        if not out.strip().lower().startswith("yes"):
            r["reasons"].append("verifier_rejected" if out else "verifier_no_reply")

    if not a.skip_verify:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(verify, todo))
    for r in rows:
        r.setdefault("status", "pass" if not r["reasons"] else "fail")
    write_jsonl(OUT / "filtered.jsonl", rows)
    from collections import Counter
    print({k: v for k, v in Counter(r["status"] for r in rows).items()})
    print("fail reasons:", dict(Counter(x for r in rows if r["status"] == "fail" for x in r["reasons"])))


if __name__ == "__main__":
    main()
