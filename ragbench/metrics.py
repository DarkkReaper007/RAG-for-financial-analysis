"""Metrics. Retrieval metrics are evidence-group aware: a question has groups G1..Gk of (doc, page) pairs; a retrieved chunk
satisfies a group if any page it spans is in the group. Any single page of a group suffices (filings repeat facts)."""
import math
import re

from ipo_rag.verification import cited_pages, numbers

ABSTAIN = ("do not contain this information", "does not contain this information", "cannot find sufficient information", "not enough information",
           "not explicitly stated", "does not provide", "not mentioned in the", "no information", "not available in the", "cannot be determined",
           "not been determined", "yet to be determined", "not determined", "unable to find", "not provided in")


def chunk_keys(c):
    return {(c["doc"], p) for p in range(c["page_start"], c["page_end"] + 1)}


def retrieval_metrics(ranked, groups, ks=(1, 3, 5, 10)):
    """ranked: chunk dicts in rank order (need doc/page_start/page_end); groups: list of sets of (doc, page)."""
    allp = set().union(*groups) if groups else set()
    keys = [chunk_keys(c) for c in ranked]
    rel = [bool(k & allp) for k in keys]
    out = {"mrr": next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)}
    for k in ks:
        sat = {gi for kk in keys[:k] for gi, g in enumerate(groups) if kk & g}
        out[f"hit@{k}"] = float(bool(sat))
        out[f"full_hit@{k}"] = float(len(sat) == len(groups))
        out[f"recall@{k}"] = len(sat) / len(groups)
        out[f"precision@{k}"] = sum(rel[:k]) / len(rel[:k]) if rel[:k] else 0.0
        seen, dcg = set(), 0.0
        for i, kk in enumerate(keys[:k]):                     # gain only for the first chunk that satisfies a group
            new = {gi for gi, g in enumerate(groups) if kk & g} - seen
            if new:
                dcg += 1.0 / math.log2(i + 2)
                seen |= new
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(groups))))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out


def _norm_fact(s):
    return re.sub(r"[\s,₹]|rs\.?|inr", "", str(s).lower())


def grounding_metrics(answer, contexts, context_pages, key_facts, answerable):
    """Deterministic checks on one answer. contexts: texts the model saw; context_pages: set of pages those contexts span."""
    abstained = any(m in answer.lower() for m in ABSTAIN)
    cited = cited_pages(answer)
    ctx_nums = numbers(" ".join(contexts), strip_cites=False)
    ans_nums = numbers(answer)
    unsupported = sorted(n for n in ans_nums if n not in ctx_nums)
    out = {"abstained": float(abstained), "has_citation": float(bool(cited)),
           "citation_valid": (sum(p in context_pages for p in cited) / len(cited)) if cited else None,
           "numeric_support": (1 - len(unsupported) / len(ans_nums)) if ans_nums else None,
           "unsupported_figure": float(bool(unsupported)) if ans_nums else None,
           "key_fact_recall": (sum(_norm_fact(f) in _norm_fact(answer) for f in key_facts) / len(key_facts)) if key_facts else None}
    out["correct_abstention"] = float(abstained) if not answerable else None
    out["false_abstention"] = float(abstained) if answerable else None
    return out
