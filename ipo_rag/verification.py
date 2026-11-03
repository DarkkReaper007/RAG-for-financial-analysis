"""Answer verification: every number in the answer must occur in the retrieved context; every cited page must have been retrieved."""
import re

CITE = re.compile(r"\bPages?:?\s*(\d+)\b", re.I)
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text, strip_cites=True):
    if strip_cites:
        text = CITE.sub(" ", text)
    out = {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}
    return {n for n in out if len(n.replace(".", "")) >= 3 or "." in n}   # skip list indices / tiny integers; keep years and amounts


def check(answer, contexts, pages):
    """contexts: list of strings; pages: pages of the retrieved contexts (all pages each context spans)."""
    ctx_nums = numbers(" ".join(contexts), strip_cites=False)
    unsupported = sorted(n for n in numbers(answer) if n not in ctx_nums)
    cited = [int(p) for p in CITE.findall(answer)]
    invalid = sorted({p for p in cited if p not in set(pages)})
    return {"unsupported_numbers": unsupported, "invalid_citations": invalid, "n_numbers": len(numbers(answer)),
            "n_citations": len(cited), "ok": not unsupported and not invalid}
