"""Corpus of persisted indexes + dense / BM25 / hybrid(RRF) retrieval, cross-encoder rerank, ordering and parent expansion."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ipo_rag.config import IngestConfig, QueryConfig
from ipo_rag.indexing import embed, index_dir, load_doc_index

_TOK = re.compile(r"[a-z0-9₹]+(?:[.,][0-9]+)*")
_rerankers = {}


def tokenize(t):
    return _TOK.findall(t.lower())


class Corpus:
    def __init__(self, cfg: IngestConfig, doc_ids, suffix=""):
        self.cfg, self.chunks, self.parents, embs = cfg, [], {}, []
        for d in doc_ids:
            chunks, parents, emb = load_doc_index(index_dir(cfg, d + suffix))
            self.chunks += chunks
            self.parents.update({f"{d}:{k}": v for k, v in parents.items()})
            for c in chunks:
                if c["parent_id"]:
                    c["parent_id"] = f"{d}:{c['parent_id']}"
            embs.append(emb)
        self.emb = np.vstack(embs)
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c["index_text"]) for c in self.chunks])
        return self._bm25


def _top(scores, n):
    idx = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
    return idx[np.argsort(-scores[idx])]


def rrf(lists, weights, k):
    score = {}
    for lst, w in zip(lists, weights):
        for rank, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + w / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def reorder_lost_in_middle(items):
    """Best chunks at the start and end of the context: [1st, 3rd, 5th ..., 6th, 4th, 2nd]."""
    return items[0::2] + items[1::2][::-1] if len(items) > 2 else items


def search(corpus: Corpus, query, q: QueryConfig):
    n = q.top_k * q.candidate_depth
    dense = bm = None
    if q.retrieval_mode in ("dense", "hybrid"):
        qv = embed(corpus.cfg.embedder, [query], "query")[0]
        dense = list(_top(corpus.emb @ qv, n))
    if q.retrieval_mode in ("bm25", "hybrid"):
        bm = list(_top(corpus.bm25.get_scores(tokenize(query)), n))
    cand = {"dense": dense, "bm25": bm}.get(q.retrieval_mode) or rrf([dense, bm], [q.rrf_dense_weight, q.rrf_sparse_weight], q.rrf_k)
    cand = [corpus.chunks[i] for i in cand[:n]]
    final = cand[:q.top_k]
    if q.rerank and cand:
        if q.reranker not in _rerankers:
            from sentence_transformers import CrossEncoder
            _rerankers[q.reranker] = CrossEncoder(q.reranker)
        sc = _rerankers[q.reranker].predict([(query, c["text"]) for c in cand])
        order = np.argsort(-sc)
        final = [{**cand[i], "rerank_score": float(sc[i])} for i in order[:q.top_k]]
    return cand, final


def assemble(corpus: Corpus, final, q: QueryConfig):
    """Final chunks -> context entries. Parent expansion (when the index has parents) + lost-in-the-middle ordering."""
    ctx, seen = [], set()
    for c in final:
        p = corpus.parents.get(c["parent_id"]) if c.get("parent_id") else None
        if p and len(p["text"]) <= q.parent_cap_chars:
            if c["parent_id"] in seen:
                continue
            seen.add(c["parent_id"])
            ctx.append({"doc": c["doc"], "text": p["text"], "page_start": p["page_start"], "page_end": p["page_end"],
                        "chapter": p["chapter"], "chunk_ids": [c["id"]], "expanded": True})
        else:
            ctx.append({"doc": c["doc"], "text": c["text"], "page_start": c["page_start"], "page_end": c["page_end"],
                        "chapter": c["chapter"], "chunk_ids": [c["id"]], "expanded": False})
    return reorder_lost_in_middle(ctx) if q.reorder else ctx
