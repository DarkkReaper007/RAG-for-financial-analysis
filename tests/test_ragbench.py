"""Model-free tests for ragbench.  python tests/test_ragbench.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ragbench import gold, metrics as M, stats as S


def ch(doc, p0, p1=None):
    return {"doc": doc, "page_start": p0, "page_end": p1 or p0}


def test_single_group_metrics():
    groups = [{("a", 5), ("a", 9)}]
    r = [ch("a", 1), ch("b", 5), ch("a", 9), ch("a", 2)]          # doc b page 5 must NOT count
    m = M.retrieval_metrics(r, groups)
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr"] == 1 / 3
    assert m["precision@4"] if "precision@4" in m else True
    assert abs(m["precision@3"] - 1 / 3) < 1e-9 and m["recall@3"] == 1.0
    assert abs(m["ndcg@3"] - 0.5) < 1e-9          # first hit at rank 3 -> 1/log2(4) / 1 = 0.5


def test_multi_group_needs_every_group():
    groups = [{("a", 5)}, {("b", 7)}]
    m = M.retrieval_metrics([ch("a", 5), ch("c", 1), ch("b", 7)], groups)
    assert m["hit@1"] == 1 and m["full_hit@1"] == 0 and m["recall@1"] == 0.5
    assert m["full_hit@3"] == 1 and m["recall@3"] == 1.0


def test_duplicate_pages_do_not_inflate_ndcg():
    groups = [{("a", 5), ("a", 6)}]
    one = M.retrieval_metrics([ch("a", 5)], groups)["ndcg@5"]
    dup = M.retrieval_metrics([ch("a", 5), ch("a", 6), ch("a", 5)], groups)["ndcg@5"]
    assert one == dup == 1.0


def test_page_span_chunk():
    assert M.retrieval_metrics([ch("a", 4, 6)], [{("a", 5)}])["hit@1"] == 1


def test_grounding():
    g = M.grounding_metrics("Price is 76 and size 9,999.00 [Acme, Page: 5] [Page: 99]", ["price 76; size 9,375.00"], {5}, ["76"], True)
    assert g["unsupported_figure"] == 1.0 and g["citation_valid"] == 0.5 and g["key_fact_recall"] == 1.0 and g["false_abstention"] == 0.0
    a = M.grounding_metrics("The provided documents do not contain this information.", ["x"], {1}, [], False)
    assert a["correct_abstention"] == 1.0 and a["numeric_support"] is None


def test_stats():
    a, b = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    d = S.paired_bootstrap(a, b)
    assert abs(d["diff"] - 0.7) < 1e-9 and d["lo"] < d["diff"] < d["hi"]
    p, n10, n01 = S.mcnemar_exact(a, b)
    assert (n10, n01) == (7, 0) and abs(p - 2 * (1 / 128)) < 1e-9
    assert S.mcnemar_exact([1, 0], [1, 0])[0] == 1.0
    h = S.holm([0.01, 0.04, 0.03])
    assert h == [0.03, 0.06, 0.06]


def test_gold_loader():
    items, h = gold.load("dev")
    assert items and all(i["split"] == "dev" for i in items) and len(h) == 64
    assert all(g for i in items if i["answerable"] for g in gold.groups_of(i))
    multi = [i for i in items if i["type"] == "multi_document"][0]
    docs = {d for g in gold.groups_of(multi) for d, _ in g}
    assert docs == set(multi["docs"])


def test_judge_scoring_and_shortcuts():
    from ragbench import judge as J
    ok = '{"coverage":{"level":"Excellent","reasoning":"r"},"faithfulness":{"level":"good","reasoning":"r"},"redundancy":{"level":"extremely bad","reasoning":"r"}} '
    s, det = J.score_reply("```json\n" + ok + "```")
    assert s == {"coverage": 1.0, "faithfulness": 0.75, "redundancy": 0.0, "overall": (1.0 + 0.75 + 0.0) / 3} and det["redundancy"]["level"] == "extremely_bad"
    assert J.score_reply('{"coverage":{"level":"great","reasoning":"r"}}') == (None, None) and J.score_reply("no json") == (None, None)
    item = {"answerable": True, "question": "q", "reference_answer": "76"}
    sc, _ = J.judge_row({"answer": "The provided documents do not contain this information."}, item, "none:model")
    assert sc == {"coverage": 0.0, "overall": 0.0}                       # abstention shortcut makes no LLM call
    assert J.judge_row({"answer": "x"}, {**item, "answerable": False}, "none:model") == (None, None)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
