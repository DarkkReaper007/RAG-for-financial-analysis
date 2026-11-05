# FinanceBench: 10-K layout labels, failure analysis, comparison with published pipelines

Subset: 52 questions over 12 filings (10-K, 10-Q, one earnings release), open corpus (all 12 filings searched). Recall = share of annotated evidence pages found.
Scripts: `ragbench/fb_sec.sh`, `ragbench/fb_diagnose.py`; layout labels: `ipo_rag/structure.py` (`RAG_STRUCTURE=sec`).

## 1. SEBI labels vs US 10-K layout labels (retrieval, n=52)

| config | SEBI labels (none detected) cand.recall@5 / final.recall@5 | 10-K layout labels cand.recall@5 / final.recall@5 |
|---|---|---|
| hybrid, section_parent (baseline) | 0.276 / 0.333 | 0.276 / 0.333 |
| dense | 0.276 / 0.314 | 0.295 / 0.285 |
| bm25 | 0.122 / 0.179 | 0.122 / 0.179 |
| section_flat | - / 0.333 | 0.276 / 0.333 |
| section_parent_ctx (section name in chunk header) | - / 0.269 | 0.346 / 0.269 |
| fixed_flat | - / 0.304 | 0.314 / 0.304 |
| oracle (search only the right filing), hybrid | 0.343 / 0.401 | 0.362 / 0.420 |

The labels change nothing for the baseline and only add +0.02 in the oracle setting. The section name in the chunk header lifts pre-rerank recall to 0.346 but
the MiniLM cross-encoder then pushes it back to 0.269, so the label helps the first stage and the reranker undoes it.

## 2. Why retrieval is weak (baseline run)

* **The filing is found, the page is not.** The right filing is in the top-5 for 88% of questions. Restricting search to the right filing (oracle) only reaches 0.40-0.42.
  Cross-filing confusion is therefore NOT the main cause (39/52 questions have at least one wrong-filing page in the top-5, 10 from a sibling 10-K/10-Q of the same company).
* **Multi-page evidence is almost impossible under this metric.** 43 questions need one page: recall 0.395. 9 questions need 2-3 pages: recall 0.0 and 0.11.
  The metric requires every annotated page group to be hit.
* **The evidence is mostly numeric statement pages.** Of the gold pages, 27 of 66 are income statement, balance sheet or cash-flow pages and 10 are notes tables.
  Such pages share near-identical text across companies and years (line-item names, no narrative), so both embeddings and BM25 find many look-alikes. 6 gold pages lie outside any detected section.
* **Many answers are computed, not looked up.** Only 22 of 39 questions with numeric key facts have those figures verbatim on the annotated pages (17 are ratios/derived).
  FinanceBench annotators list the pages used as inputs, but the question text ("is 3M capital intensive?") does not resemble those pages.
* **Page-level scoring undercounts valid retrieval.** For the 22 extractive questions, the answer figures appear in the retrieved top-5 pages for 18 (9 of the 13 that count as misses).
  Number string matching can be coincidental for short figures, so treat this as an upper bound, but it shows alternate valid pages exist (the same figure appears in MD&A and in the statements).
* **Near misses are rare:** only 5/35 missed questions retrieved a page within 2 pages of an annotated page.
* **Retrieval is the bottleneck for answers:** nano with the gold pages scores judge.overall 0.594 vs 0.344 with retrieved pages.
* **Subset effects.** n=52, so the 95% interval on recall is about +-0.12; 3M 10-K, AES and PepsiCo (very long filings, 250-500 pages) score 0.0, Amcor/Boeing/3M 10-Q score 0.57-0.75.
* The pipeline was tuned for SEBI documents (hybrid, chapter chunking, MiniLM reranker, small English embedder); nothing was tuned for 10-K statements.

## 3. Comparison with published FinanceBench pipelines

These are answer-accuracy figures on the 150-question open set and are NOT directly comparable to our page recall or judge score (different subset, metric and grader). Use them as context only.

| pipeline (source) | setup | reported correct |
|---|---|---|
| GPT-4-Turbo, shared vector store (Islam et al. 2023, FinanceBench paper) | one index over all filings, ada-002 | 19% (68% refused) |
| Llama 2, shared vector store (same) | same | 19.3% |
| GPT-4-Turbo, per-document vector store (same) | index per filing | 50% (39% refused) |
| Ragie, shared store top_k=8 / single store top_k=32 (vendor blog) | no rerank | 27% / 51% |
| GPT-4-Turbo + oracle page (same paper, via secondary summary) | gold evidence given | ~85% |
| this pipeline, open corpus (52-question subset) | hybrid+rerank, nano generator | judge.overall 0.34 (key-fact recall 0.15) |
| this pipeline, gold pages given (ceiling) | nano | judge.overall 0.59 |

What lines up with the literature: shared-store (open corpus) setups are much worse than per-document ones, and in this work restricting to the right filing raises recall from 0.33 to 0.40-0.42.
Literature on financial-report chunking (arXiv 2402.05131) reports element-based chunking helps; our section/fixed variants differ by less than the noise.
We did not run those external systems ourselves, so no head-to-head comparison is claimed.
