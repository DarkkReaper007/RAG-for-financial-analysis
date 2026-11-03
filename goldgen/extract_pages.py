"""Stage 0: raw page text for every filing (independent of the RAG pipeline's parser).

    python goldgen/extract_pages.py [--docs zomato ola_electric]
Writes goldgen/out/pages/<doc>.jsonl: page (1-indexed PDF page), text, chars, numeric_ratio, table_like, section.
"""
import argparse, re, sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import OUT, ROOT, manifest, write_jsonl

NUM = re.compile(r"\d[\d,\.]*")

from ipo_rag.structure import chapter_of


def section_map(doc, n):
    """page -> deepest bookmark title at/before the page (DRHPs usually carry a bookmark outline)."""
    toc = doc.get_toc()  # [level, title, page]
    m, cur = {}, None
    by_page = {}
    for lvl, title, pg in toc:
        by_page.setdefault(pg, []).append((lvl, title.strip()))
    for p in range(1, n + 1):
        if p in by_page:
            cur = by_page[p][-1][1]
            top = by_page[p][0][1] if by_page[p][0][0] == 1 else None
        m[p] = cur
    return m, len(toc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    a = ap.parse_args()
    (OUT / "pages").mkdir(exist_ok=True)
    for did, d in manifest().items():
        if a.docs and did not in a.docs:
            continue
        doc = fitz.open(ROOT / "data" / "prospectuses" / d["file"])
        smap, ntoc = section_map(doc, len(doc))
        rows, chap = [], None
        for i, page in enumerate(doc, 1):
            t = page.get_text()
            lines = [l for l in t.splitlines() if l.strip()]
            numeric = sum(1 for l in lines if len(NUM.findall(l)) >= 2) / max(1, len(lines))
            c = chapter_of(t)
            if c and (i > 8 or c in ("DEFINITIONS AND ABBREVIATIONS",)):
                chap = c
            rows.append({"doc": did, "page": i, "text": t, "chars": len(t), "numeric_ratio": round(numeric, 3),
                         "table_like": numeric > 0.35 and len(lines) > 12, "toc_section": smap[i], "chapter": chap})
        write_jsonl(OUT / "pages" / f"{did}.jsonl", rows)
        print(f"{did:16} pages={len(rows):4d} toc_entries={ntoc:4d} table_like={sum(r['table_like'] for r in rows):4d} "
              f"empty={sum(r['chars'] < 100 for r in rows)} chapters={len({r['chapter'] for r in rows if r['chapter']})} "
              f"labelled={sum(r['chapter'] is not None for r in rows)/len(rows):.0%}")


if __name__ == "__main__":
    main()
