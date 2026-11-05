"""Why is FinanceBench retrieval weak? Breaks the baseline run down by evidence spread, question type, document confusion and gold-page section.
  RAG_CORPUS=financebench RAG_GOLD=fb1 python -m ragbench.fb_diagnose [run_dir_name]"""
import collections
import json
import sys

import fitz

from ipo_rag.corpus import manifest, pdf_path
from ipo_rag.structure import label_pages_sec
from ragbench import gold
from ragbench.runner import RESULTS

run = sys.argv[1] if len(sys.argv) > 1 else "3_retrieval_mode__hybrid__ext__retrieval"
items = {i["id"]: i for i in gold.load("ext")[0]}
rows = [json.loads(l) for l in open(RESULTS / run / "rows.jsonl")]
docs = manifest()
labels = {d: label_pages_sec([p.get_text() for p in fitz.open(pdf_path(docs[d]))]) for d in {x for i in items.values() for x in i["docs"]}}


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def table(title, groups):
    print(f"\n### {title}\n\n| group | n | final.recall@5 | final.full_hit@5 | right-doc in top-5 |\n|---|---|---|---|---|")
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        print(f"| {k} | {len(rs)} | {mean([r['final']['recall@5'] for r in rs])} | {mean([r['final']['full_hit@5'] for r in rs])} | {mean([r['_doc5'] for r in rs])} |")


for r in rows:
    it = items[r["id"]]
    top5 = r["retrieved"][:5]
    r["_doc5"] = float(any(t.split(":")[0] in it["docs"] for t in top5))
    r["_wrongdoc"] = [t.split(":")[0] for t in top5 if t.split(":")[0] not in it["docs"]]
ans = [r for r in rows if items[r["id"]]["answerable"]]
table("All", {"all": ans})
table("Number of evidence groups (pages that must all be found)", collections.defaultdict(list, {k: v for k, v in [(g, [r for r in ans if len(items[r["id"]]["evidence_groups"]) == g]) for g in (1, 2, 3, 4)] if v}))
table("FinanceBench question type", {t: [r for r in ans if items[r["id"]]["type"] == t] for t in {items[r["id"]]["type"] for r in ans}})
table("Filing", {d: [r for r in ans if items[r["id"]]["docs"][0] == d] for d in sorted({items[r["id"]]["docs"][0] for r in ans})})

# sections the gold pages sit in
sec = collections.Counter()
for r in ans:
    it = items[r["id"]]
    for g in it["evidence_groups"]:
        for pg in g:
            sec[(labels[it["docs"][0]][pg - 1] or "unlabelled")] += 1
print("\n### Section of the gold evidence pages (SEC layout labels)\n\n| section | gold pages |\n|---|---|")
for k, v in sec.most_common():
    print(f"| {k} | {v} |")

# confusion with sibling filings of the same company
conf = sum(1 for r in ans if r["_wrongdoc"])
sib = sum(1 for r in ans if any(w.split("_")[0] == items[r["id"]]["docs"][0].split("_")[0] for w in r["_wrongdoc"]))
print(f"\nAnswerable questions with at least one top-5 page from a wrong filing: {conf}/{len(ans)}; from a sibling filing of the same company (10-K vs 10-Q): {sib}")

# are the key facts even on the gold pages (retrieval can only help for extractive answers)?
ext = comp = 0
for r in ans:
    it = items[r["id"]]
    txt = " ".join(fitz.open(pdf_path(docs[it["docs"][0]]))[pg - 1].get_text() for g in it["evidence_groups"] for pg in g).replace(",", "")
    kf = [k for k in it["key_facts"] if any(c.isdigit() for c in k)]
    if kf:
        if all(k.replace(",", "") in txt for k in kf):
            ext += 1
        else:
            comp += 1
print(f"Questions whose numeric key facts all appear verbatim on the gold pages: {ext}; computed/derived (not verbatim): {comp}")

# near miss: gold page missed but a page within +-2 of it was retrieved
near = 0
miss = [r for r in ans if r["final"]["recall@5"] < 1]
for r in miss:
    it = items[r["id"]]
    gp = {pg for g in it["evidence_groups"] for pg in g}
    got = set()
    for t in r["retrieved"][:5]:
        d, pg = t.split(":")
        a, _, b = pg.partition("-")
        got |= {(d, n) for n in range(int(a), int(b or a) + 1)}
    if any((it["docs"][0], pg + d) in got for pg in gp for d in (-2, -1, 1, 2)):
        near += 1
print(f"Questions with a miss where a page within 2 pages of an evidence page was retrieved: {near}/{len(miss)}")

# alternate evidence: for extractive questions, does the retrieved context contain the answer even when it is not the annotated page?
alt = tot = alt_miss = tot_miss = 0
for r in ans:
    it = items[r["id"]]
    kf = [k.replace(",", "") for k in it["key_facts"] if any(c.isdigit() for c in k)]
    if not kf:
        continue
    gtxt = " ".join(fitz.open(pdf_path(docs[it["docs"][0]]))[pg - 1].get_text() for g in it["evidence_groups"] for pg in g).replace(",", "")
    if not all(k in gtxt for k in kf):
        continue                                            # derived answer, skip
    ptxt = ""
    for t in r["retrieved"][:5]:
        d, pg = t.split(":")
        a, _, b = pg.partition("-")
        for n in range(int(a), int(b or a) + 1):
            ptxt += fitz.open(pdf_path(docs[d]))[n - 1].get_text()
    ptxt = ptxt.replace(",", "")
    ok = all(k in ptxt for k in kf)
    tot += 1; alt += ok
    if r["final"]["recall@5"] < 1:
        tot_miss += 1; alt_miss += ok
print(f"Extractive questions (answer verbatim on the annotated pages): {tot}; answer text present in the retrieved top-5 pages: {alt} ({alt_miss} of the {tot_miss} counted as misses)")
