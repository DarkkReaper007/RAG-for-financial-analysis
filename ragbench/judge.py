"""LLM-as-judge: compares the GOLD answer with the PREDICTED answer only (no retrieved context), one call per answer.

Three axes, each on a 5-level scale (excellent 1.0 / good 0.75 / acceptable 0.5 / bad 0.25 / extremely_bad 0.0), with the judge's reasoning kept verbatim:
  coverage     - are all gold items present in the prediction?
  faithfulness - are the predicted items accurate (none wrong) relative to the gold answer?   [NOT context-faithfulness]
  redundancy   - is each item stated once, without filler?
overall = mean of the three. Answerable questions answered with an abstention get coverage 0 and overall 0 without an LLM call (faithfulness and
redundancy are not defined for an empty answer, but the miss must count against the mean). Unanswerable items are scored by the deterministic abstention checks, not here.

  python -m ragbench.judge --run verify_off__dev__generation [--judge groq:openai/gpt-oss-120b] [--tag groq]
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from ragbench import gold
from ragbench.llm import chat
from ragbench.metrics import ABSTAIN
from ragbench.runner import RESULTS, aggregate

LEVELS = {"excellent": 1.0, "good": 0.75, "acceptable": 0.5, "bad": 0.25, "extremely_bad": 0.0}
AXES = ("coverage", "faithfulness", "redundancy")

PROMPT = """You are grading a predicted answer against a gold (reference) answer for a question about an Indian IPO offer document.
Compare ONLY the two answers. Do not use outside knowledge.

QUESTION: {q}

GOLD ANSWER:
{gold}

PREDICTED ANSWER:
{pred}

First split the GOLD ANSWER into its distinct items (a single value or fact is one item; a list has one item per element).
Treat spacing, casing, spelling variants, reordering, citations like [Company, Page: N], and numerically equal values written in different
units or formats (e.g. "₹1,000 million" and "₹1 billion", "18.66%" and "18.66 percent") as the SAME item.

Grade three axes. Use exactly one level per axis: excellent, good, acceptable, bad, extremely_bad.

coverage - are all gold items present in the prediction?
  excellent: none missing | good: only 1-2 missing (or at least 75% present) | acceptable: more than 1-2 missing but at least half present |
  bad: fewer than half present, but at least one | extremely_bad: none present (all missing)

faithfulness - are the items stated in the prediction accurate relative to the gold answer, with none wrong or invented?
  A predicted item is wrong if it contradicts the gold answer or gives different information for the same question.
  A predicted item that is NOT in the gold answer counts as invented (wrong), unless it is only a rewording, a citation, or a restatement of the question.
  excellent: none wrong | good: only 1-2 wrong | acceptable: more than 1-2 wrong but at least half correct |
  bad: fewer than half correct, but at least one | extremely_bad: all wrong

redundancy - is each item stated once, with no filler?
  excellent: every item appears once, no filler | good: only 1-2 redundant items | acceptable: noticeable repetition but at least half of the statements are unique |
  bad: fewer than half unique (mostly duplicates or filler) | extremely_bad: entirely redundant, no new information

If the gold answer has fewer than 5 items, judge by the share of items instead of counting 1-2 (e.g. 1 of 2 items missing is "acceptable").
Reply with JSON only, with a one-sentence reasoning per axis:
{{"coverage": {{"level": "...", "reasoning": "..."}}, "faithfulness": {{"level": "...", "reasoning": "..."}}, "redundancy": {{"level": "...", "reasoning": "..."}}}}"""


def jparse(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def score_reply(reply):
    """Judge reply text -> ({axis: score}, {axis: {level, reasoning}}) or (None, None) if unusable."""
    j = jparse(reply)
    if not isinstance(j, dict):
        return None, None
    scores, detail = {}, {}
    for ax in AXES:
        e = j.get(ax)
        lvl = str((e or {}).get("level", "")).strip().lower().replace(" ", "_").replace("-", "_") if isinstance(e, dict) else ""
        if lvl not in LEVELS:
            return None, None
        scores[ax], detail[ax] = LEVELS[lvl], {"level": lvl, "reasoning": str(e.get("reasoning", ""))}
    scores["overall"] = sum(scores[a] for a in AXES) / len(AXES)
    return scores, detail


def judge_pair(question, gold_answer, pred, model, max_tokens=1500):
    prompt = PROMPT.format(q=question, gold=gold_answer, pred=pred[:3000])
    if os.environ.get("RAG_CORPUS", "ipo") != "ipo":
        prompt = prompt.replace("an Indian IPO offer document", "a company financial filing (SEC 10-K, 10-Q or earnings release)")
    reply = chat(model, prompt, max_tokens=max_tokens)
    return score_reply(reply)


def judge_row(row, item, model):
    """-> (scores | None, detail | None). Abstention on an answerable item: coverage 0, no LLM call."""
    if not item["answerable"]:
        return None, None
    pred = row["answer"]
    if any(m in pred.lower() for m in ABSTAIN):
        return {"coverage": 0.0, "overall": 0.0}, {"coverage": {"level": "extremely_bad", "reasoning": "Prediction abstains; no gold item is present."}}
    return judge_pair(item["question"], item["reference_answer"], pred, model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under ragbench/results (generation mode)")
    ap.add_argument("--judge", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for output files so several judges can be compared on one run")
    a = ap.parse_args()
    d = RESULTS / a.run
    run = json.load(open(d / "run.json"))
    if a.judge.split(":", 1)[-1] == run["config"]["gen"]["model"]:
        print("WARNING: judge == generator (self-grading bias)", file=sys.stderr)
    items = {i["id"]: i for i in gold.load(run["split"])[0]}
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    todo = [r for r in rows if not r["failed"]]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: judge_row(r, items[r["id"]], a.judge), todo))
    unjudged = 0
    for r, (s, det) in zip(todo, res):
        if s is not None:
            r["judge"], r["judge_detail"] = s, det
        elif items[r["id"]]["answerable"]:
            unjudged += 1                       # judge gave no usable reply; left out rather than guessed
    suffix = f".{a.tag}" if a.tag else ""
    with open(d / f"rows.judged{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = aggregate(rows)
    json.dump({"judge_model": a.judge, "unjudged": unjudged, **agg}, open(d / f"metrics.judged{suffix}.json", "w"), indent=1)
    m = agg["overall"]["metrics"]
    print(a.run, "judge:", a.judge, "unjudged:", unjudged, {k: round(v["mean"], 3) for k, v in m.items() if k.startswith("judge.")})


if __name__ == "__main__":
    main()
