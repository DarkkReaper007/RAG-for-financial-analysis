"""Stage 2: one question per sampled page with a hosted LLM (key from NVIDIA_API_KEY).
    python goldgen/gen_questions.py [--model nvidia/nemotron-3.5-bolt-e4b] [--workers 4]"""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import ALIASES, OUT, chat, manifest, parse_json, read_jsonl, write_jsonl

PROMPT = """You are helping build an evaluation set for question answering over Indian IPO offer documents (DRHP/RHP).
Below is ONE page (PDF page {page}) from the {doc_type} of {company}. Write ONE {target} question that a retail investor or analyst could ask, answerable ONLY from this page.

Rules:
- Mention the company naturally ("{short}'s ...", "at {short}", "in {short}'s offer document") - never append "for {short}" at the end. For financial figures also name the fiscal year or date.
- Prefer what an investor or analyst would care about (business, financials, risks, offer terms, governance, litigation, use of proceeds) over incidental detail such as street addresses or minor administrative dates.
- Write a natural, complete question of at least 8 words.
- Paraphrase. Do not copy any phrase of 5 or more consecutive words from the page.
- The question must make sense without seeing the page: never say "this page", "the table", "above", "below" or "the document".
- The answer must be stated explicitly on the page and be short and factual. If the needed value is a placeholder such as [●], or the page is an index, boilerplate or only headings, reply {{"skip": true, "reason": "..."}}.
- "quote": copy ONE contiguous span (at most 300 characters) from the page that contains the answer, EXACTLY as written, including numbers and punctuation.
Type guide: definition = ask what a defined term or abbreviation means in this document; factoid = a name, date, count or yes/no fact; numeric = a specific amount, percentage or ratio; table = read one value (or compare two rows) from a table on the page; list = ask for a set of 3-6 items listed on the page.

PAGE:
\"\"\"
{text}
\"\"\"

Reply with JSON only: {{"question": "...", "answer": "...", "quote": "...", "answer_type": "{target}"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-3.5-bolt-e4b")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    man = manifest()
    pages = {}
    for did in {s["doc"] for s in read_jsonl(OUT / "sample.jsonl")}:
        pages[did] = {r["page"]: r for r in read_jsonl(OUT / "pages" / f"{did}.jsonl")}

    def work(s):
        m, text = man[s["doc"]], pages[s["doc"]][s["page"]]["text"][:7000]
        raw = chat(a.model, PROMPT.format(page=s["page"], doc_type="Prospectus" if m["doc_type"] == "prospectus" else "Draft Red Herring Prospectus",
                                          company=m["company"].title(), short=ALIASES[s["doc"]][0], target=s["target_type"], text=text),
                   temperature=0.3, max_tokens=500)
        j = parse_json(raw) or {}
        return {**s, "gen_model": a.model, "raw_ok": bool(j), "skip": bool(j.get("skip")) or not j,
                "question": j.get("question"), "answer": j.get("answer"), "quote": j.get("quote"), "skip_reason": j.get("reason")}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(work, read_jsonl(OUT / "sample.jsonl")))
    for i, r in enumerate(rows):
        r["cid"] = f"{r['doc']}-p{r['page']}"
    write_jsonl(OUT / "candidates.jsonl", rows)
    print(f"{len(rows)} pages -> {sum(not r['skip'] for r in rows)} candidates, {sum(r['skip'] for r in rows)} skipped")


if __name__ == "__main__":
    main()
