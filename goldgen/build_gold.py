"""Assemble the gold set.  python goldgen/build_gold.py  ->  eval/gold/gold_v1.json (+ sha256)
Inputs: out/evidence.jsonl (single-page items with evidence groups), out/special.jsonl (multi-doc + unanswerable)."""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import OUT, ROOT, manifest, read_jsonl
from ipo_rag.verification import NUM

TYPE_MAP = {"factoid": "factoid", "numeric": "numeric", "table": "table", "list": "list", "definition": "definition"}


def key_facts(answer):
    a = str(answer).strip()
    nums = [n for n in NUM.findall(a) if len(n.replace(",", "").replace(".", "")) >= 3 or "." in n]
    if nums:
        return nums
    return [a] if len(a.split()) <= 6 else []


def tidy(q):
    q = re.sub(r"\s+", " ", q).strip()
    q = q[0].upper() + q[1:]
    return q if q.endswith("?") else q.rstrip(".") + "?"


def main():
    man = manifest()
    items, seq = [], Counter()

    def add(base):
        split = man[base["docs"][0]]["split"]
        seq[(split, base["type"])] += 1
        items.append({"id": f"{split[:1]}-{base['type'][:3]}-{seq[(split, base['type'])]:02d}", "split": split, **base})

    for r in read_jsonl(OUT / "evidence.jsonl"):
        add({"type": TYPE_MAP[r["target_type"]], "docs": [r["doc"]], "topic": (r["chapter"] or "").title(), "question": tidy(r["question"]),
             "answerable": True, "evidence_groups": r["evidence_groups"], "reference_answer": str(r["answer"]),
             "key_facts": key_facts(r["answer"]), "tags": {"lexical_overlap": r["tags"]["lexical_overlap"], "needs_table": r["target_type"] == "table",
                                                              "spread": "single_page"},
             "provenance": {"method": "llm_generated", "generator": r["gen_model"], "source_page": r["page"], "quote": r["quote"],
                            "ratings": {"natural": r.get("natural"), "value": r.get("value")}, "verification": ["llm_verifier", "spot_check_by_claude"]}})
    for r in read_jsonl(OUT / "special.jsonl"):
        if r["status"] != "ok":
            continue
        if r["kind"] == "multidoc":
            add({"type": "multi_document", "docs": r["docs"], "topic": "General Information", "question": r["question"], "answerable": True,
                 "evidence_groups": r["evidence_groups"], "reference_answer": r["reference_answer"], "key_facts": [s["answer"] for s in r["sides"] if len(s["answer"].split()) <= 6],
                 "tags": {"lexical_overlap": None, "needs_table": False, "spread": "multi_document"},
                 "provenance": {"method": "llm_extracted_template", "attribute": r["attr"], "quotes": [s["quote"] for s in r["sides"]],
                                "verification": ["llm_verifier", "quote_on_page"]}})
        else:
            add({"type": "unanswerable", "docs": r["docs"], "topic": r["attr"], "question": r["question"], "answerable": False, "evidence_groups": [],
                 "reference_answer": r["reference_answer"], "key_facts": [], "tags": {"lexical_overlap": None, "needs_table": False, "spread": "none"},
                 "provenance": {"method": "template", "template": r["attr"], "verification": ["by_construction"]}})
    gold = {"version": "v1", "note": "LLM-generated and LLM-verified (independent verifier model); spot-checked by Claude (see goldgen/REVIEW_LOG.md). NOT human-verified.",
            "items": items}
    d = ROOT / "eval" / "gold"
    d.mkdir(exist_ok=True)
    body = json.dumps(gold, indent=1, ensure_ascii=False)
    (d / "gold_v1.json").write_text(body)
    (d / "gold_v1.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    print(len(items), "items;", dict(Counter(i["split"] for i in items)), dict(Counter(i["type"] for i in items)))


if __name__ == "__main__":
    main()
