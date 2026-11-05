"""Pure, LLM-free metric functions (unit-tested in test_metrics.py)."""
import math
import random
import re
from statistics import mean

CITE_RE = re.compile(r"\bPages?:?\s*(\d+)\b", re.I)  # bracketed or bare "Page: 357"
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
ABSTAIN_MARKERS = ("cannot find sufficient information", "not enough information", "not explicitly stated",
                   "does not provide", "not mentioned in the", "no information", "not available in the",
                   "cannot be determined", "do not contain", "not been determined", "yet to be determined", "not determined")


def is_rel(page, gold_pages, tol=0):
    return any(abs(int(page) - g) <= tol for g in gold_pages)


def hit_at_k(pages, gold, k, tol=0):
    return float(any(is_rel(p, gold, tol) for p in pages[:k]))


def precision_at_k(pages, gold, k, tol=0):
    top = pages[:k]
    return sum(is_rel(p, gold, tol) for p in top) / len(top) if top else 0.0


def recall_at_k(pages, gold, k, tol=0):
    """Share of gold pages covered by the top-k results."""
    if not gold:
        return 0.0
    return sum(any(abs(int(p) - g) <= tol for p in pages[:k]) for g in gold) / len(gold)


def mrr(pages, gold, tol=0):
    for i, p in enumerate(pages, 1):
        if is_rel(p, gold, tol):
            return 1.0 / i
    return 0.0


def ndcg_at_k(pages, gold, k, tol=0):
    rels = [1.0 if is_rel(p, gold, tol) else 0.0 for p in pages[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def retrieval_scores(pages, gold, ks=(1, 3, 5, 10), tol=0):
    out = {"mrr": mrr(pages, gold, tol)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(pages, gold, k, tol)
        out[f"recall@{k}"] = recall_at_k(pages, gold, k, tol)
        out[f"precision@{k}"] = precision_at_k(pages, gold, k, tol)
        out[f"ndcg@{k}"] = ndcg_at_k(pages, gold, k, tol)
    return out


def _norm_num(s):
    return s.replace(",", "").rstrip(".")


def extract_numbers(text):
    """Numbers worth checking: >=3 digits or containing a decimal point (skips list indices, years are kept)."""
    text = CITE_RE.sub(" ", text)
    nums = {_norm_num(m) for m in NUM_RE.findall(text)}
    return {n for n in nums if len(n.replace(".", "")) >= 3 or "." in n}


def numeric_support(answer, contexts):
    """Fraction of numbers in the answer that appear in the context (hallucinated-figure check). None if no numbers."""
    nums = extract_numbers(answer)
    if not nums:
        return None
    ctx = extract_numbers(" ".join(contexts))
    return sum(n in ctx for n in nums) / len(nums)


def citation_accuracy(answer, retrieved_pages):
    """Fraction of cited pages that were actually retrieved. None if the answer has no citations."""
    cited = [int(p) for p in CITE_RE.findall(answer)]
    if not cited:
        return None
    pages = {int(p) for p in retrieved_pages}
    return sum(c in pages for c in cited) / len(cited)


def has_citation(answer):
    return bool(CITE_RE.search(answer))


def is_abstention(answer):
    a = answer.lower()
    return any(m in a for m in ABSTAIN_MARKERS)


def _fact_norm(s):
    return re.sub(r"[\s,₹]|rs\.?|inr", "", s.lower())


def fact_recall(answer, key_facts):
    """Fraction of required key facts present in the answer (normalised substring match)."""
    if not key_facts:
        return None
    a = _fact_norm(answer)
    return sum(_fact_norm(f) in a for f in key_facts) / len(key_facts)


def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = (len(xs) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def mean_ci(xs, n_boot=1000, seed=0):
    """(mean, lo, hi) with a 95% bootstrap CI; (mean, None, None) when n<5."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    m = mean(xs)
    if len(xs) < 5:
        return m, None, None
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
