"""Re-check the frozen gold set: schema, doc/page bounds, quotes on pages, hash, distribution.   python goldgen/validate_gold.py"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import OUT, ROOT, manifest, norm, read_jsonl

man, errs = manifest(), []
body = (ROOT / "eval" / "gold" / "gold_v1.json").read_text()
if hashlib.sha256(body.encode()).hexdigest() != (ROOT / "eval" / "gold" / "gold_v1.sha256").read_text().strip():
    errs.append("hash mismatch")
G = json.loads(body)["items"]
pages = {}
ids = Counter(i["id"] for i in G)
for i in G:
    if ids[i["id"]] > 1: errs.append(f"{i['id']}: duplicate id")
    for d in i["docs"]:
        if d not in man: errs.append(f"{i['id']}: unknown doc {d}")
        elif man[d]["split"] != i["split"]: errs.append(f"{i['id']}: split mismatch")
        pages.setdefault(d, {p["page"]: norm(p["text"]) for p in read_jsonl(OUT / "pages" / f"{d}.jsonl")})
    if i["answerable"] != bool(i["evidence_groups"]): errs.append(f"{i['id']}: answerable/evidence mismatch")
    if i["type"] == "multi_document" and len(i["evidence_groups"]) != len(i["docs"]): errs.append(f"{i['id']}: groups != docs")
    for gi, grp in enumerate(i["evidence_groups"]):
        d = i["docs"][gi] if i["type"] == "multi_document" else i["docs"][0]
        for p in grp:
            if p not in pages[d]: errs.append(f"{i['id']}: page {p} out of range for {d}")
    q = i["provenance"].get("quote")
    if q and norm(q) not in pages[i["docs"][0]].get(i["provenance"]["source_page"], ""): errs.append(f"{i['id']}: quote not on source page")
    for gi, q in enumerate(i["provenance"].get("quotes", [])):
        if norm(q) not in pages[i["docs"][gi]][i["evidence_groups"][gi][0]]: errs.append(f"{i['id']}: multidoc quote {gi} not on page")
    if not i["question"].strip().endswith("?"): errs.append(f"{i['id']}: question lacks '?'")
print(f"{len(G)} items, {len(errs)} problems")
for e in errs[:40]: print(" ", e)
print("by split:", dict(Counter(i["split"] for i in G)))
print("by type: ", dict(Counter(i["type"] for i in G)))
print("by doc:  ", dict(sorted(Counter(d for i in G for d in i["docs"]).items())))
print("evidence pages per answerable item (mean):", round(sum(sum(len(g) for g in i["evidence_groups"]) for i in G if i["answerable"]) / max(1, sum(i["answerable"] for i in G)), 2))
sys.exit(1 if errs else 0)
