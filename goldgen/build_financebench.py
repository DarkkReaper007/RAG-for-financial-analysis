"""Convert the FinanceBench open set (subset of filings) into our manifest + gold format.
  python goldgen/build_financebench.py
Evidence pages are located by matching the evidence text on the real PDF pages (FinanceBench page numbers are not reliable indexing)."""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FB = ROOT / "data" / "financebench"
DOCS = ["AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K", "AMCOR_2023_10K", "ULTABEAUTY_2023Q4_EARNINGS", "3M_2022_10K",
        "3M_2023Q2_10Q", "AES_2022_10K", "BESTBUY_2023_10K", "BESTBUY_2024Q2_10Q", "CVSHEALTH_2022_10K"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    Q = [json.loads(l) for l in open(FB / "financebench_open_source.jsonl")]
    Q = [q for q in Q if q["doc_name"] in DOCS]
    info = {}
    for q in Q:
        info.setdefault(q["doc_name"], q)
    pages, man = {}, []
    for d in DOCS:
        pdf = fitz.open(FB / "pdfs" / f"{d}.pdf")
        pages[d] = [norm(p.get_text()) for p in pdf]
        q = info[d]
        kind = "10-K" if d.endswith("10K") else "10-Q" if "10Q" in d else "earnings release"
        man.append({"id": d, "file": f"{d}.pdf", "company": q["company"].upper(), "doc_type": kind, "sector": q.get("gics_sector") or "", "year": int(re.search(r"(20\d\d)", d).group(1)),
                    "split": "ext", "pages": len(pdf), "sha256": hashlib.sha256((FB / "pdfs" / f"{d}.pdf").read_bytes()).hexdigest()})
    json.dump({"version": 1, "source": "patronus-ai/financebench (open-source subset)", "docs": man}, open(FB / "manifest.json", "w"), indent=1)

    def find_pages(doc, text):
        """Pages of `doc` containing the evidence: try several distinctive windows of the evidence text."""
        t = norm(text)
        words = t.split()
        wins = [" ".join(words[i:i + 8]) for i in range(0, max(1, len(words) - 8), max(1, len(words) // 6))][:6]
        wins = [w for w in wins if len(w) > 25] or [t[:60]]
        hit = collections.Counter()
        for pi, pt in enumerate(pages[doc], 1):
            for w in wins:
                if w in pt:
                    hit[pi] += 1
        if not hit:
            return []
        best = max(hit.values())
        return sorted(p for p, c in hit.items() if c == best)

    items, unmatched = [], 0
    for q in Q:
        groups = []
        for ev in q["evidence"]:
            pgs = find_pages(q["doc_name"], ev["evidence_text"]) or find_pages(q["doc_name"], ev.get("evidence_text_full_page", "")[:400])
            if not pgs:                                            # last resort: the published page number, 0- or 1-indexed
                pgs = [min(len(pages[q["doc_name"]]), ev["evidence_page_num"] + 1)]
                unmatched += 1
            groups.append(pgs)
        ans = str(q["answer"])
        nums = [n for n in re.findall(r"\d[\d,]*(?:\.\d+)?", ans) if len(n.replace(",", "").replace(".", "")) >= 2]
        kf = [n.replace(",", "").rstrip("0").rstrip(".") if "." in n else n.replace(",", "") for n in nums][:3]
        items.append({"id": q["financebench_id"].replace("financebench_id_", "fb-"), "split": "ext", "type": q["question_type"].replace("-", "_"),
                      "docs": [q["doc_name"]], "topic": q.get("question_reasoning") or "", "question": q["question"], "answerable": True, "evidence_groups": groups,
                      "reference_answer": ans, "key_facts": kf, "tags": {"lexical_overlap": None, "needs_table": None, "spread": "single_page" if len(groups) == 1 else "multi_evidence"},
                      "provenance": {"method": "financebench_open_source", "financebench_id": q["financebench_id"], "justification": q.get("justification"), "verification": ["published_benchmark"]}})
    body = json.dumps({"version": "fb1", "note": "FinanceBench open-source subset (12 filings). Evidence pages located by text match on the PDFs.", "items": items}, indent=1, ensure_ascii=False)
    (ROOT / "eval" / "gold" / "gold_fb1.json").write_text(body)
    (ROOT / "eval" / "gold" / "gold_fb1.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    print(f"{len(items)} items over {len(DOCS)} filings ({sum(m['pages'] for m in man)} pages); evidence located by text match for "
          f"{sum(len(i['evidence_groups']) for i in items) - unmatched}/{sum(len(i['evidence_groups']) for i in items)} evidence items; by type {dict(collections.Counter(i['type'] for i in items))}")
    print("evidence pages per group:", dict(collections.Counter(len(g) for i in items for g in i["evidence_groups"])))


if __name__ == "__main__":
    main()
