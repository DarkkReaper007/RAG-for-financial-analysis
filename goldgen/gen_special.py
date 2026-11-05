"""Multi-document and unanswerable items.   python goldgen/gen_special.py  ->  goldgen/out/special.jsonl
Multi-doc: one attribute extracted from two filings of the SAME split (quote must be on the page; verifier confirms each side).
Unanswerable: questions a pre-listing offer document cannot answer, by construction (post-listing price, future fiscal year,
final price in a draft that still shows [●]); the [●] case is checked against the filing."""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldgen.common import DISPLAY, OUT, chat, manifest, norm, parse_json, read_jsonl, write_jsonl

ATTR = {  # attribute: (single-doc sub-question, keywords, multi-doc question template, chapter hint)
    "auditor": ("Who is the statutory auditor of {c}?", ["statutory auditor"], "Who are the statutory auditors of {a} and of {b}?"),
    "registrar": ("Who is the registrar to the offer for the IPO of {c}?", ["registrar to the offer"], "Who is the registrar to the offer in the IPOs of {a} and {b}?"),
    "facevalue": ("What is the face value of one equity share of {c}?", ["face value"], "What is the face value of one equity share of {a} and of {b}, and are they the same?"),
    "office": ("In which city or locality is the registered office of {c} located?", ["registered office"], "In which city or locality are the registered offices of {a} and {b} located?"),
}
PAIRS = [("zomato", "delhivery", "auditor"), ("lic", "ola_electric", "registrar"), ("ola_electric", "manipal_health", "facevalue"),
         ("zomato", "manipal_health", "office"), ("delhivery", "lic", "registrar"),                                  # dev
         ("nykaa", "swiggy", "auditor"), ("hyundai_india", "lg_india", "registrar"), ("hdb_financial", "nse", "office"),
         ("jio_platforms", "swiggy", "facevalue"), ("nykaa", "hyundai_india", "office"), ("lg_india", "jio_platforms", "auditor"),
         ("hdb_financial", "nykaa", "registrar")]                                                                    # test
UNANS = {"zomato": "post_listing", "nykaa": "post_listing", "swiggy": "post_listing", "lic": "price_tbd", "ola_electric": "future_fy",
         "hyundai_india": "price_tbd", "hdb_financial": "future_fy", "lg_india": "price_tbd", "delhivery": "future_fy",
         "manipal_health": "price_tbd", "jio_platforms": "future_fy", "nse": "price_tbd"}
UQ = {"post_listing": "What was the closing price of the shares of {c} on their first day of trading on the stock exchanges?",
      "future_fy": "What was the revenue from operations of {c} in Fiscal 2030?",
      "price_tbd": "What is the final offer price per equity share in the IPO of {c}?"}
EXTRACT = """PAGE:
\"\"\"
{page}
\"\"\"
Question: {q}
If the PAGE states the answer, reply with JSON {{"found": true, "answer": "<short answer>", "quote": "<contiguous span of at most 250 characters copied EXACTLY from the PAGE>"}}. Otherwise reply {{"found": false}}."""
VERIFY = """PAGE:
\"\"\"
{page}
\"\"\"
QUESTION: {q}
CANDIDATE ANSWER: {a}
Can the question be answered from the PAGE, and is the candidate answer correct according to the PAGE? Reply with exactly one word: yes or no."""


def find(doc_id, kws, pages, n=3):
    scored = [(sum(norm(p["text"]).count(k) for k in kws) + (2 if p["chapter"] == "GENERAL INFORMATION" else 0), p) for p in pages if p["page"] < 200]
    return [p for s, p in sorted(scored, key=lambda x: -x[0])[:n] if s > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="nvidia/nemotron-3.5-bolt-e4b")
    ap.add_argument("--verifier", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    a = ap.parse_args()
    man = manifest()
    P = {d: read_jsonl(OUT / "pages" / f"{d}.jsonl") for d in man}

    def side(did, attr):
        sub, kws, _ = ATTR[attr]
        q = sub.format(c=DISPLAY[did])
        for p in find(did, kws, P[did]):
            j = parse_json(chat(a.gen, EXTRACT.format(page=p["text"][:7000], q=q), temperature=0, max_tokens=400)) or {}
            if j.get("found") and j.get("quote") and norm(j["quote"]) in norm(p["text"]):
                v = chat(a.verifier, VERIFY.format(page=p["text"][:7000], q=q, a=j["answer"]), temperature=0, max_tokens=4000)
                if v.strip().lower().startswith("yes"):
                    return {"doc": did, "page": p["page"], "answer": j["answer"], "quote": j["quote"]}
        return None

    def multi(t):
        d1, d2, attr = t
        s1, s2 = side(d1, attr), side(d2, attr)
        if not (s1 and s2):
            return {"kind": "multidoc", "docs": [d1, d2], "attr": attr, "status": "failed", "sides": [bool(s1), bool(s2)]}
        c1, c2 = DISPLAY[d1], DISPLAY[d2]
        return {"kind": "multidoc", "docs": [d1, d2], "attr": attr, "status": "ok", "question": ATTR[attr][2].format(a=c1, b=c2),
                "reference_answer": f"{c1}: {s1['answer']}. {c2}: {s2['answer']}.", "evidence_groups": [[s1["page"]], [s2["page"]]], "sides": [s1, s2]}

    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(multi, PAIRS))
    for did, kind in UNANS.items():
        ok = True
        if kind == "price_tbd":     # confirm the draft still shows the price as a placeholder
            ok = any("●" in p["text"] and "price" in p["text"].lower() for p in P[did][:120])
        out.append({"kind": "unanswerable", "docs": [did], "attr": kind, "status": "ok" if ok else "failed",
                    "question": UQ[kind].format(c=DISPLAY[did]),
                    "reference_answer": {"post_listing": "Not in the offer document: it was filed before listing.",
                                         "future_fy": "Not in the offer document: it covers historical periods only.",
                                         "price_tbd": "Not determined in the draft offer document (shown as [●])."}[kind]})
    write_jsonl(OUT / "special.jsonl", out)
    print({k: sum(1 for r in out if r["kind"] == k and r["status"] == "ok") for k in ("multidoc", "unanswerable")}, "failed:", [(r["docs"], r["attr"]) for r in out if r["status"] != "ok"])


if __name__ == "__main__":
    main()
