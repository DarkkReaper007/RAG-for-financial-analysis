"""Find candidate PDF pages for gold labelling (no index/LLM needed).

    python eval/label_helper.py --doc 1626944616560.pdf --find "price band" "floor price"
Prints 1-indexed pages ranked by keyword hits, with a snippet. Verify by eye, then fill relevant_pages / key_facts.
"""
import argparse, re, fitz

ap = argparse.ArgumentParser()
ap.add_argument("--doc", required=True)
ap.add_argument("--find", nargs="+", required=True)
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()
d = fitz.open(f"data/prospectuses/{a.doc}")
hits = []
for i, p in enumerate(d, 1):
    t = p.get_text()
    n = sum(len(re.findall(re.escape(k), t, re.I)) for k in a.find)
    if n:
        m = re.search(re.escape(a.find[0]), t, re.I) or re.search("|".join(map(re.escape, a.find)), t, re.I)
        s = max(0, m.start() - 80)
        hits.append((n, i, " ".join(t[s:s + 240].split())))
for n, i, snip in sorted(hits, reverse=True)[: a.top]:
    print(f"page {i:4d}  hits={n:3d}  ...{snip}...")
