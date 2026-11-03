# RAG Evaluation: Research, Plan, and Before/After Comparison

## Part 1 — How RAG Systems Are Evaluated (Research Summary)

RAG evaluation decomposes into three layers: **retrieval quality**, **generation quality**,
and **end-to-end/operational** metrics. Sources are cited at the bottom of each section.

### 1.1 Retrieval Evaluation (classic IR metrics)

These require a labeled set of (query → relevant document/chunk IDs) pairs and score the
retriever independent of generation:

| Metric | What it measures | Order-aware? |
|---|---|---|
| **Precision@K** | Fraction of the top-K retrieved chunks that are actually relevant | No |
| **Recall@K** | Fraction of *all* relevant chunks that were found in the top-K | No |
| **Hit Rate@K** | Did *at least one* relevant chunk appear in the top-K? (binary, per query) | No |
| **MRR** (Mean Reciprocal Rank) | 1/rank of the *first* relevant hit — rewards surfacing a good chunk early | Yes |
| **nDCG@K** | Rewards ranking highly-relevant chunks above marginally-relevant ones, discounted by position | Yes |

These are cheap, deterministic, and don't need an LLM — but they require ground-truth
relevance labels, which is the hard part in practice.

### 1.2 Generation Evaluation

Given a question, the retrieved context, and the generated answer, judge the *answer* itself:

- **Faithfulness / Groundedness** — are the answer's claims actually supported by the
  retrieved context (not the model's outside knowledge)? The core hallucination check.
- **Answer Relevance** — does the answer actually address the question asked (not just
  fluent, on-topic-adjacent text)?
- **Answer Correctness** — factual overlap between the generated answer and a *reference*
  (ground-truth) answer — requires labeled Q&A pairs, unlike faithfulness/relevance which
  don't need a reference answer.

### 1.3 The RAG Triad (TruLens / TruEra)

A minimal 3-metric framework that needs no ground-truth labels, each scored by an
LLM-as-judge:

1. **Context Relevance** — is each retrieved chunk relevant to the query? (retrieval-side)
2. **Groundedness** — is the answer supported by the retrieved context? (same idea as
   faithfulness above)
3. **Answer Relevance** — does the answer address the query? (generation-side)

Passing all three gives confidence the system isn't hallucinating *and* isn't just
retrieving noise. This is the same idea RAGAS formalizes with more granular metrics.

### 1.4 RAGAS Framework

The most widely adopted open-source RAG evaluation library. Core metrics:

- **Context Precision** — proportion of retrieved chunks that were actually *useful* for
  generating the answer (penalizes retrieving irrelevant padding)
- **Context Recall** — did the retrieved context contain everything needed to fully answer
  the question? (requires a reference answer to check against)
- **Faithfulness** — same concept as groundedness above
- **Answer Relevancy** — same concept as the triad's answer relevance
- The **RAGAS score** is the mean of these four

This project's `src/evaluation/llm_evaluator.py` (built this session) implements a
subset of this: faithfulness, answer relevance, context precision via LLM-as-judge, plus a
simple keyword-based context recall.

### 1.5 More Advanced / Recent Frameworks

- **ARES** (Stanford) — fine-tunes small classifier models on synthetic data to predict
  context relevance / faithfulness / answer relevance cheaply, with statistical confidence
  intervals (PPI) instead of raw LLM-judge calls every time.
- **RAGChecker** — decomposes both the reference answer *and* the generated answer into
  atomic "claims," then checks entailment between them — gives claim-level diagnostics
  (precision/recall of individual facts) instead of one holistic score per question.
- **RAGBench** — a large multi-domain benchmark dataset (4,162 questions across 10 domains)
  for training/testing RAG evaluators themselves.

### 1.6 Operational / System Metrics

Not about answer quality, but essential in practice:
- **Latency** — retrieval time vs. generation time, broken out separately (this pipeline
  already reports both — see `pipeline.py`'s `timing` dict)
- **Cost/throughput** — tokens processed, $ per query (for hosted APIs) or compute-seconds
  (for self-hosted)
- **Retrieval-vs-generation attribution** — when an answer is bad, was it a *retrieval*
  failure (right info wasn't found) or a *generation* failure (right info was found but the
  model didn't use it correctly)? RAGChecker and the RAG Triad are both explicitly designed
  to separate these two failure modes.

**Sources:**
- [Complete Guide to RAG Evaluation — GetMaxim](https://www.getmaxim.ai/articles/complete-guide-to-rag-evaluation-metrics-methods-and-best-practices-for-2025/)
- [RAG Evaluation Metrics — Confident AI](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [Evaluating RAG using RAGAS — Superlinked](https://superlinked.com/blog/evaluating-retrieval-augmented-generation-ragas)
- [RAG Triad — TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [What is the RAG Triad? — TruEra](https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/)
- [Building and Evaluating Advanced RAG — DeepLearning.AI](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/lawtd/rag-triad-of-metrics)
- [Evaluation Measures in Information Retrieval — Pinecone](https://www.pinecone.io/learn/offline-evaluation/)
- [How to Evaluate Retrieval Quality: DCG@k and NDCG@k — Towards Data Science](https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-part-3-dcgk-and-ndcgk/)
- [Evaluation Metrics for Search and Recommendation — Weaviate](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [ARES — Stanford FutureData (GitHub)](https://github.com/stanford-futuredata/ARES)
- [RAGChecker — GetMaxim](https://www.getmaxim.ai/blog/ragchecker-eval-tool/)
- [Answer Correctness — Ragas docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)
- [RAGBench — arXiv](https://arxiv.org/pdf/2407.11005)

---

## Part 2 — Evaluation Suite Plan for This Project

### Current state (what exists today)

| Layer | Implemented? | Where |
|---|---|---|
| Retrieval-only IR metrics (Hit Rate, MRR, nDCG, Precision/Recall@K) | ❌ Not implemented | — |
| Context Precision (LLM-judged) | ✅ | `llm_evaluator.py` |
| Context Recall | ⚠️ Keyword-only, not LLM-judged, needs `expected_sections` | `evaluator.py` / `llm_evaluator.py` |
| Faithfulness / Groundedness | ✅ (rule-based *and* LLM-judge) | `evaluator.py`, `llm_evaluator.py` |
| Answer Relevance | ✅ (rule-based *and* LLM-judge) | `evaluator.py`, `llm_evaluator.py` |
| Answer Correctness (vs. reference answer) | ❌ Not implemented — no reference answers exist | — |
| Operational (latency breakdown) | ✅ | `pipeline.py` (`retrieval_seconds`, `generation_seconds`) |
| Independent judge (not self-grading) | ❌ Judge = same model as generator | `llm_evaluator.py` |

### Gaps and proposed additions, in priority order

**1. Gold evaluation set (blocking — everything else benefits from this)**
Build a small labeled set per document: for ~10 questions, hand-annotate (a) which
chunk IDs/sections are actually relevant, and (b) a short reference answer. Without this,
Context Recall and Answer Correctness can't be computed honestly, and retrieval metrics
have nothing to score against. This is manual work but only needs to happen once per
document and pays for every future evaluation run.

**2. Retrieval-only evaluation mode**
Add a method that calls `HybridRetriever.retrieve()` directly (skipping generation
entirely) and scores Hit Rate@K, MRR, and Recall@K against the gold relevant-chunk labels.
This is the cheapest metric to compute (no LLM calls, pure Python) and isolates retrieval
quality from generation quality — critical for knowing *what to fix* when scores are low.

**3. Independent judge option**
Currently the judge is the same model as the generator (self-grading — a known bias risk,
already flagged in this session). Add an option to pass a *different* `RAGGenerator`
instance as `llm_judge=` — e.g. judge with Llama 3.2 while generating with Mistral, or
vice versa — so self-grading bias can be measured and avoided in production.

**4. Answer Correctness (needs the gold set from #1)**
Once reference answers exist, add a metric comparing the generated answer to the
reference — via LLM-judged semantic similarity, which is simpler to build here than full
RAGAS-style claim decomposition (RAGChecker-style) and still useful.

**5. RAG Triad framing as the headline dashboard**
Rather than exposing 5+ raw metric numbers, group the existing metrics under the three
triad labels for the Gradio Evaluation tab: **Context Relevance** (= context precision),
**Groundedness** (= faithfulness), **Answer Relevance** (already named this). Simple UI
change, no new computation — makes the dashboard match the industry-standard framing so
the numbers can be talked about externally.

**6. Retrieval/generation failure attribution**
When faithfulness or relevance is low, currently there's no automatic signal for *why*.
Adding context-relevance-per-chunk (not just averaged) would let the evaluator flag "this
answer was ungrounded because none of the 5 retrieved chunks were relevant" (retrieval
failure) vs. "the chunks were relevant but the answer ignored them" (generation failure) —
this is the core idea behind RAGChecker and the RAG Triad's 3-way split.

### What this session's `benchmark_evaluation.py` already covers vs. the full plan

`benchmark_evaluation.py` currently runs the *existing* rule-based and LLM-judge
evaluators back-to-back on the same 10 sample questions and prints a diff table. It's a
good harness for #3 and #5 above (add a judge param, group by triad) but doesn't yet do
retrieval-only scoring (#2) or correctness (#4) since those need the gold set (#1) first.

---

## Part 3 — Changes Made This Session

1. **Configuration system** (new) — `src/config/loader.py` + `config/models.yaml`: model
   name/URL/temperature/timeout are now config-driven (YAML + env var overrides) instead of
   hardcoded in `app.py`.
2. **Model upgrade: Llama 3.2 3B → Mistral 7B** (default) — free, open-weight, same Ollama
   API so it works identically self-hosted on a free-tier cloud VM.
3. **LLM-as-judge evaluator** (new) — `src/evaluation/llm_evaluator.py`: semantic
   faithfulness/relevance/context-precision scoring via LLM calls, replacing/supplementing
   the rule-based n-gram/keyword evaluator. Selectable per-run via a `HybridEvaluator`.
4. **`RAGGenerator.raw_complete()`** (new method) — lets the judge send prompts without the
   financial-analyst system prompt/citation template corrupting scoring output.
5. **Evaluation mode UI toggle** — Gradio Evaluation tab now has a Rule-Based / LLM-Judge
   radio selector; `RAGPipeline.set_evaluator_mode()` swaps evaluators without rebuilding
   the whole pipeline.
6. **Judge reuses the pipeline's own generator** — no duplicate model connection (though
   this means self-grading, flagged as a gap in Part 2).
7. **`benchmark_evaluation.py`** (new) — CLI script to run both evaluators back-to-back and
   print a comparison table.
8. **Bug fixes found via actual testing** (not just code review):
   - Judge prompts were being corrupted by the answer-generation template (fixed via `raw_complete`)
   - `HybridEvaluator` crashed with `AttributeError` in LLM-judge mode
   - Evaluation-mode toggle silently no-op'd after first pipeline init
   - Local generation timeout regressed 120s→60s during config migration, then found even
     120s insufficient for a cold-start 7B model — settled on 180s based on measured data
9. **Documentation** — `IMPROVEMENTS.md`, `QUICKSTART.md`, this file.

---

## Part 4 — Before/After Comparison (measured, limited)

Setup: one document (Zomato, 420 pages, 2077 chunks), 2 questions, rule-based evaluator, Apple-silicon laptop (no dedicated GPU), ChromaDB fallback (Docker/Qdrant unavailable).

**BEFORE (Llama 3.2 3B, pre-session code path)**

| Metric | Value |
|---|---|
| Ingest | 227.8s |
| Q1 (risk factors) | **timed out at 240s** (error string returned as the answer) |
| Q2 (issue size / price band) | answered, with citations; correctly said total issue size not in context |
| Avg generation | 143.2s (inflated by timeout) |
| Rule-based faithfulness / relevance / ctx precision / hit rate | 0.046 / 0.6 / 1.0 / 1.0 |

The 0.046 faithfulness is an artefact: the timeout error text was scored as an answer. The evaluator has no notion of "generation failed". Load average was ~8 at the time, so timing is noisy.

**AFTER (Mistral 7B + LLM-judge):** the run completed but its results file was lost when the session was forked, so **no AFTER numbers are claimed here**. Re-run to fill this in:

```bash
python benchmark_evaluation.py
```

**Takeaways (supported by the data)**
- Generation latency on local hardware is the dominant risk; both models can exceed 240s under load. Add retries and mark failed generations as errors instead of scoring them.
- Earlier uncontrolled runs had Llama 3.2 at ~100s for Q1; run-to-run variance is large, so compare over more questions before drawing conclusions.
- Self-grading (judge == generator) remains an unquantified bias.
