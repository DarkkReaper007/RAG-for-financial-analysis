"""Stage 1: stratified page sample per filing.   python goldgen/sample_pages.py --docs zomato ola_electric --per-doc 20"""
import argparse, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import OUT, manifest, read_jsonl, write_jsonl

SKIP_CHAPTERS = {None, "DEFINITIONS AND ABBREVIATIONS", "FORWARD-LOOKING STATEMENTS", "DECLARATION", "DIVIDEND POLICY_",
                 "MAIN PROVISIONS OF THE ARTICLES OF ASSOCIATION", "MATERIAL CONTRACTS AND DOCUMENTS FOR INSPECTION"}
TYPE_CYCLE = ["factoid", "numeric", "list"]


def eligible(r):
    t = r["text"].lower()
    return (r["page"] > 10 and r["chars"] >= 900 and r["chapter"] not in SKIP_CHAPTERS
            and "intentionally left blank" not in t[:600] and t.count("[●]") < 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--per-doc", type=int, default=20)
    ap.add_argument("--table-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--definition-docs", nargs="*", default=[], help="docs that also get one definitions-chapter page")
    a = ap.parse_args()
    out = []
    for did in manifest():
        if a.docs and did not in a.docs:
            continue
        rng = random.Random(f"{a.seed}-{did}")
        pages = [r for r in read_jsonl(OUT / "pages" / f"{did}.jsonl") if eligible(r)]
        tab = [r for r in pages if r["table_like"]]
        rest = [r for r in pages if not r["table_like"]]
        n_tab = min(len(tab), round(a.per_doc * a.table_share))
        chosen = rng.sample(tab, n_tab)
        by_ch = {}
        for r in rest:
            by_ch.setdefault(r["chapter"], []).append(r)
        for v in by_ch.values():
            rng.shuffle(v)
        order = list(by_ch)
        rng.shuffle(order)
        while len(chosen) < a.per_doc and any(by_ch.values()):       # round-robin across chapters
            for ch in order:
                if by_ch[ch] and len(chosen) < a.per_doc:
                    chosen.append(by_ch[ch].pop())
        chosen.sort(key=lambda r: r["page"])
        for k, r in enumerate(chosen):
            out.append({"doc": did, "page": r["page"], "chapter": r["chapter"],
                        "target_type": "table" if r["table_like"] else TYPE_CYCLE[k % len(TYPE_CYCLE)]})
        if did in a.definition_docs:
            dp = [r for r in read_jsonl(OUT / "pages" / f"{did}.jsonl") if r["chapter"] == "DEFINITIONS AND ABBREVIATIONS" and r["chars"] > 1800 and not r["table_like"] and r["page"] > 10]
            if dp:
                r = rng.choice(dp)
                out.append({"doc": did, "page": r["page"], "chapter": r["chapter"], "target_type": "definition"})
        print(f"{did:16} eligible={len(pages):4d} sampled={len(chosen):3d} chapters={len({r['chapter'] for r in chosen})}")
    write_jsonl(OUT / "sample.jsonl", out)


if __name__ == "__main__":
    main()
