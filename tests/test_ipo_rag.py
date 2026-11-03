"""Model-free unit tests.  python tests/test_ipo_rag.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace

from ipo_rag.chunking import build_chunks
from ipo_rag.config import ABLATION_LEVELS, BASELINE, IngestConfig, with_changes
from ipo_rag.parsing import table_text
from ipo_rag.retrieval import reorder_lost_in_middle, rrf
from ipo_rag.structure import chapter_of, label_pages
from ipo_rag.verification import check

DOC = {"id": "x", "company": "ACME LIMITED", "doc_type": "drhp"}


def pages(n=6, per=5):
    out = []
    for p in range(1, n + 1):
        out.append({"page": p, "text": "", "chapter": "RISK FACTORS" if p <= 3 else "OUR BUSINESS",
                    "blocks": [f"Paragraph {p}.{j} " + "word " * 40 for j in range(per)]})
    return out


def test_chapter_detection():
    assert chapter_of("28\nSECTION II: RISK FACTORS\nAn investment...") == "RISK FACTORS"
    assert chapter_of("RISK FACTORS\nOUR BUSINESS\nINDUSTRY OVERVIEW\nGENERAL INFORMATION") is None   # contents page
    assert label_pages(["x"] * 10 + ["SECTION V: FINANCIAL INFORMATION"], 9)[-1] == "FINANCIAL INFORMATION"


def test_section_chunks_respect_chapters_and_size():
    cfg = IngestConfig(table_mode="inline", child_chars=500, child_overlap=50)
    chunks, parents = build_chunks(DOC, pages(), [], cfg)
    kids = [c for c in chunks if c["type"] == "child"]
    assert kids and all(len(c["text"]) <= 500 + 60 for c in kids)
    assert all(c["chapter"] == parents[c["parent_id"]]["chapter"] for c in kids)       # parent never spans chapters
    assert {c["chapter"] for c in kids} == {"RISK FACTORS", "OUR BUSINESS"}
    assert not any(c["chapter"] == "RISK FACTORS" and c["page_start"] > 3 for c in kids)


def test_fixed_chunks_and_flat_hierarchy():
    cfg = replace(IngestConfig(table_mode="inline"), chunker="fixed", hierarchy="flat", child_chars=400, child_overlap=40)
    chunks, parents = build_chunks(DOC, pages(), [], cfg)
    assert parents == {} and all(c["parent_id"] is None for c in chunks)
    assert all(len(c["text"]) <= 400 for c in chunks) and chunks[0]["page_start"] == 1


def test_contextual_header_and_tables():
    tbl = [{"page": 2, "rows": [["Item", "Value"], ["Price", "76"], ["Size", "9,375"]], "caption": "Offer details"}]
    chunks, _ = build_chunks(DOC, pages(), tbl, replace(IngestConfig(), contextual_header=True, table_mode="markdown"))
    t = [c for c in chunks if c["type"] == "table"][0]
    assert t["index_text"].startswith("[Acme | DRHP | Risk Factors | p.2]") and "| Item | Value |" in t["text"] and "Table: Offer details" in t["text"]
    assert table_text(tbl[0], "flat") == "Item | Value\nPrice | 76\nSize | 9,375"
    inline, _ = build_chunks(DOC, pages(), tbl, replace(IngestConfig(), table_mode="inline"))
    assert not any(c["type"] == "table" for c in inline)


def test_fusion_and_ordering():
    assert rrf([[1, 2, 3], [3, 2, 9]], [0.5, 0.5], 60)[:2] == [2, 3] or rrf([[1, 2, 3], [3, 2, 9]], [0.5, 0.5], 60)[0] in (2, 3)
    assert reorder_lost_in_middle([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]


def test_verification():
    ctx = ["Offer price is Rs. 76 per share; issue size 9,375.00 million."]
    ok = check("Price is 76 and size 9,375.00 million [Acme, Page: 5]", ctx, {5, 6})
    assert ok["ok"] and ok["n_citations"] == 1
    bad = check("Size is 10,000.00 million [Acme, Page: 99]", ctx, {5})
    assert bad["unsupported_numbers"] == ["10000.00"] and bad["invalid_citations"] == [99]


def test_config():
    assert with_changes(BASELINE, chunker="fixed", retrieval_mode="dense", model="m").query.retrieval_mode == "dense"
    assert BASELINE.ingest.key() != with_changes(BASELINE, chunker="fixed").ingest.key()
    assert BASELINE.ingest.key() == with_changes(BASELINE, retrieval_mode="dense").ingest.key()   # query-time knobs don't change the index
    for study, levels in ABLATION_LEVELS.items():
        for name, kw in levels.items():
            with_changes(BASELINE, **kw)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
