# Evaluation suite

Layered, branch-independent. Raw runs and judging are separate steps, so the judge is held constant across arms.

| Layer | What | Needs LLM | Needs labels |
|---|---|---|---|
| 1 Ingestion | page coverage, company-name check, empty chunks, chunk size p5/p50/p95, tables/sections | no | company names only |
| 2 Retrieval (ablation) | dense vs BM25 vs hybrid(RRF) vs hybrid+rerank: hit/recall/precision/nDCG@k, MRR | no | `relevant_pages` |
| 3 Grounding (deterministic) | citation present + citation accuracy (cited page was retrieved), numeric support (answer numbers found in context), key-fact recall, correct/false abstention | no | `key_facts`, `answerable` |
| 4 RAG triad (LLM judge) | context relevance per chunk, claim-level faithfulness, answer relevance, correctness vs reference | yes (fixed judge) | `reference_answer` for correctness |
| 5 System | failure rate (timeouts), generation latency p50/p95, ingest time | no | no |

Reporting: means with 95% bootstrap CIs (n>=5), per-category faithfulness.

## Run
```bash
# from the repo root of the code under test (git worktree per arm)
python eval/test_metrics.py                                   # unit tests, no model needed
python eval/label_helper.py --doc 1626944616560.pdf --find "price band"   # find pages to label
python eval/run_benchmark.py --label C_mistral --model mistral:7b --mode all     # or: ingestion | retrieval | generation
python eval/judge.py eval/results/C_mistral.json --judge-model qwen2.5-coder:7b-instruct-q4_K_M
python eval/compare.py eval/results/*.judged.json
```
Arms to compare: A `main`+Llama 3.2, B feature+Llama 3.2 (code-only effect), C feature+Mistral 7B. Run each 2-3 times (local latency is noisy).

## Caveats
- Metrics that need labels are skipped for unlabelled questions; the report prints `n=` everywhere. Fill `gold_questions.json` (30+ items, all 4 docs, plus multi-document and unanswerable cases).
- Page-level relevance: tables spanning pages may need `--tol 1`.
- Retrieval ablation depth uses `retrieve(top_k=depth)`; reranking sees 2x depth candidates, so full-pipeline numbers differ slightly from `top_k=5` generation runs.
- Judge must not be a generator under test (self-grading bias); a warning is printed otherwise. Small judges are noisy: spot-check 10 judgements by hand.
- The index is rebuilt per run (~4 min per large PDF); retrieval mode skips generation so it is cheap after ingest.
- Failed generations are excluded from quality metrics and reported as failure rate.
