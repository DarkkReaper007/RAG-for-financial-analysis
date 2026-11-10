"""Smoke test for a running demo server: many questions x every setting, with timings.
  python -m demo.server                      # in one terminal (set NVIDIA_API_KEYS there to test answer writing)
  python -m demo.smoke_test                  # in another: quick matrix
  python -m demo.smoke_test --full           # every mode x verify x hierarchy x filing scope for every question (long with a key)
  python -m demo.smoke_test --fresh          # ignore saved answers so generation time is real
Prints one line per call, then a summary by setting. Exit code 1 if any call returned an HTTP error or an unexpected failure."""
import argparse
import itertools
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = [  # (label, question, filings)  filings: None = all 12
    ("fact", "Who is the registrar to the offer?", ["swiggy"]),
    ("table", "What was the profit for the year achieved by Hyundai in Fiscal 2022 compared to Fiscal 2021?", ["hyundai_india"]),
    ("number", "What is the minimum bid amount required for Retail Individual Bidders?", ["lic"]),
    ("list", "Who are the statutory auditors of the company?", ["zomato"]),
    ("definition", "What does the abbreviation MCA signify within Zomato's offer document?", ["zomato"]),
    ("two filings", "Who is the registrar to the offer in the IPOs of Hyundai Motor India and LG Electronics India?", ["hyundai_india", "lg_india"]),
    ("all filings", "Which company has a book running lead manager called Kotak Mahindra Capital?", None),
    ("not in document", "What is the final offer price per equity share in the IPO of Hyundai Motor India?", ["hyundai_india"]),
    ("vague", "risks", ["nykaa"]),
    ("one word", "IPO", None),
    ("long", "Please explain in detail, with all the relevant numbers and dates that you can find, how the company intends to use the net proceeds of the fresh issue and what portion is allocated to each of the stated objects of the offer", ["ola_electric"]),
    ("odd characters", "What is the P/E ratio? <b>\"quotes\"</b> & ampersand ₹ 100 crore", ["delhivery"]),
]


def call(base, path, body=None, timeout=300):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, raw, time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t
    except Exception as e:
        return 0, str(e).encode(), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="refresh=true: bypass saved answers")
    ap.add_argument("--out", default="", help="write all rows as JSON here")
    a = ap.parse_args()

    code, raw, _ = call(a.base, "/api/status")
    if code != 200:
        print("server not reachable:", raw[:200]); return 2
    st = json.loads(raw)
    print(f"server ready={st['ready']} answers_live={st['answers_live']} model={st['model']} saved_answers={st['saved_answers']}")
    while not st["ready"]:
        time.sleep(2); st = json.loads(call(a.base, "/api/status")[1])

    if a.full:
        settings = [dict(mode=m, verify=v, hierarchy=h) for m, v, h in itertools.product(("hybrid", "dense", "bm25"), ("off", "on"), ("parent", "flat"))]
    else:
        settings = [dict(mode="hybrid", verify="off", hierarchy="parent"), dict(mode="dense", verify="off", hierarchy="parent"), dict(mode="bm25", verify="off", hierarchy="parent"),
                    dict(mode="hybrid", verify="on", hierarchy="parent"), dict(mode="hybrid", verify="off", hierarchy="flat")]
    rows, bad = [], 0
    for (label, q, docs), s in itertools.product(QUESTIONS, settings):
        body = dict(question=q, docs=docs or [], refresh=a.fresh, **s)
        code, raw, wall = call(a.base, "/api/ask", body)
        row = dict(label=label, **s, scope=len(docs) if docs else 12, http=code, wall=round(wall, 2))
        if code == 200:
            d = json.loads(raw)
            t = d["timing"]
            row.update(failed=d["failed"], kind=d.get("error_kind"), cached=d["cached"], abstained=d["abstained"], retrieval=t["retrieval_s"], generation=t["generation_s"],
                       passages=len(d["stages"]["reranked"]), answer_chars=len(d["answer"]), citations=len(d["citations"]))
            ok_fail = d["failed"] and d.get("error_kind") == "no_key" and not st["answers_live"]       # expected when the server has no key
            if d["failed"] and not ok_fail:
                bad += 1
            if not d["stages"]["reranked"]:
                row["problem"] = "no passages"; bad += 1
            cites = [c for c in d["citations"] if c["doc"] and c["pages"]]
            if cites and not d["failed"]:                                           # the cited page image must load
                c0, _, _ = call(a.base, f"/api/page/{cites[0]['doc']}/{cites[0]['pages'][0]}")
                if c0 != 200:
                    row["problem"] = f"cited page returned {c0}"; bad += 1
        else:
            row["error"] = raw[:160].decode(errors="replace"); bad += 1
        rows.append(row)
        print(f"{label:16} {s['mode']:6} verify={s['verify']:3} {s['hierarchy']:6} http={code} wall={row['wall']:6.2f}s"
              f"  search={row.get('retrieval')}s answer={row.get('generation')}s  {'FAILED ' + str(row.get('kind')) if row.get('failed') else ''}{row.get('problem', '')}{row.get('error', '')}", flush=True)

    print("\nSummary by setting (seconds, median / max wall time)")
    for s in settings:
        r = [x["wall"] for x in rows if all(x[k] == s[k] for k in s) and x["http"] == 200]
        if r:
            print(f"  {s['mode']:6} verify={s['verify']:3} {s['hierarchy']:6} n={len(r):2}  median={statistics.median(r):6.2f}  max={max(r):6.2f}")
    gens = [x["generation"] for x in rows if x.get("generation")]
    if gens:
        print(f"  answer writing: median {statistics.median(gens):.1f} s, max {max(gens):.1f} s over {len(gens)} answers")
    print(f"\n{len(rows)} calls, {bad} problems")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
