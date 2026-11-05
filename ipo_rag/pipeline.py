"""RAGPipeline: ingest (cached per config) -> retrieve -> assemble -> generate -> verify."""
import time

from ipo_rag.config import BASELINE, PipelineConfig
from ipo_rag.corpus import manifest
from ipo_rag.generation import format_context, generate
from ipo_rag.indexing import build_doc_index
from ipo_rag.retrieval import Corpus, assemble, search
from ipo_rag.verification import check

ABSTAIN = "The provided documents do not contain this information."


class RAGPipeline:
    def __init__(self, cfg: PipelineConfig = BASELINE, docs=None, max_pages=None, log=print):
        self.cfg, self.man, self.log, self.max_pages = cfg, manifest(), log, max_pages
        self.doc_ids = docs or list(self.man)
        self.corpus = None

    def ingest(self, force=False):
        for d in self.doc_ids:
            build_doc_index(self.man[d], self.cfg.ingest, self.max_pages, force, self.log)
        self.corpus = Corpus(self.cfg.ingest, self.doc_ids, suffix=f"__first{self.max_pages}" if self.max_pages else "")
        return self

    def retrieve(self, question, doc_ids=None):
        """Retrieval only (no LLM): returns pre-rerank candidates, final chunks and assembled contexts."""
        if self.corpus is None:
            self.ingest()
        t = time.time()
        corpus = self.corpus
        if doc_ids:
            corpus = _subset(self.corpus, doc_ids)
        cand, final = search(corpus, question, self.cfg.query)
        return {"candidates": cand, "final": final, "contexts": assemble(corpus, final, self.cfg.query), "retrieval_s": round(time.time() - t, 3)}

    def answer(self, question, doc_ids=None):
        return self.answer_from(question, self.retrieve(question, doc_ids))

    def answer_from(self, question, r):
        """Generate (and optionally verify) from an existing retrieval result; safe to call from several threads."""
        ctxs = r["contexts"]
        names = {d: self.man[d]["company"].title() for d in self.man}
        block = format_context(ctxs, names)
        g = generate(self.cfg.gen, question, block)
        out = {"question": question, **{k: r[k] for k in ("candidates", "final", "contexts", "retrieval_s")}, "gen": g,
               "answer": g["answer"], "failed": g["failed"], "verification": None, "verification_action": None}
        if self.cfg.query.verify == "on" and not g["failed"] and g["answer"] != ABSTAIN:
            pages = {p for c in ctxs for p in range(c["page_start"], c["page_end"] + 1)}
            v = check(g["answer"], [c["text"] for c in ctxs], pages)
            out["verification"] = v
            if not v["ok"]:
                fb = (f"Your previous answer contained figures not found in the context ({', '.join(v['unsupported_numbers']) or 'none'}) "
                      f"or citations to pages not provided ({v['invalid_citations'] or 'none'}). Rewrite the answer using only figures and pages that appear in the context, or say the information is not available.")
                g2 = generate(self.cfg.gen, question, block, extra=fb)
                v2 = check(g2["answer"], [c["text"] for c in ctxs], pages) if not g2["failed"] else v
                out["verification_retry"] = {"answer": g2["answer"], "check": v2}
                if not g2["failed"] and v2["ok"]:
                    out["answer"], out["verification_action"] = g2["answer"], "regenerated_ok"
                elif v["unsupported_numbers"]:
                    out["answer"], out["verification_action"] = ABSTAIN + f" (unverifiable figures: {', '.join(v['unsupported_numbers'][:5])})", "abstained"
                else:
                    out["verification_action"] = "citation_only_flag"
        return out


def _subset(corpus, doc_ids):
    """View of a Corpus restricted to some docs (shares arrays; BM25 re-built for the subset)."""
    import copy
    import numpy as np
    keep = [i for i, c in enumerate(corpus.chunks) if c["doc"] in set(doc_ids)]
    sub = copy.copy(corpus)
    sub.chunks, sub.emb, sub._bm25 = [corpus.chunks[i] for i in keep], corpus.emb[np.array(keep)], None
    return sub
