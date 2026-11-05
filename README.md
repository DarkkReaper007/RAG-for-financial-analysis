# RAG for Indian IPO offer documents (DRHP / RHP / Prospectus)

Config-driven RAG pipeline + gold-set generation + evaluation suite, built to support an ablation study.
Work branch: `final`. The older prototype lives in `src/` and `app.py` (kept for reference, not used by the new code).

## Layout
| Path | Purpose |
|---|---|
| `ipo_rag/` | **New pipeline.** `config.py` (every ablation factor), `parsing`, `chunking`, `indexing` (persisted, exact cosine), `retrieval`, `generation`, `verification`, `pipeline`, `cli` |
| `goldgen/` | LLM-assisted gold-set generation (complete; see `goldgen/README.md`) |
| `eval/gold/` | `gold_v1.json` + sha256 |
| `ragbench/` | **Evaluation stack and experiment runner** (metrics, judge, ablation orchestrator, final stage, pooled analysis); raw per-question results in `ragbench/results/` (not committed); curated reports in `results/` |
| `results/` | **Committed result reports** (`SUMMARY.md` is the index; FinanceBench in `report__financebench_analysis.md`) |
| `review_app/` | Gradio UI to verify the gold set against the PDF pages |
| `eval/` | Legacy scaffold from the prototype phase + `eval/gold/` (the gold set) |
| `data/manifest.json` | 12 filings, fixed dev (5) / test (7) split by document, sha256 of each PDF |
| `tests/` | Model-free unit tests: `python tests/test_ipo_rag.py` |

PDFs (`data/prospectuses/`, `data/financebench/pdfs/`) and built indexes (`indexes/`) are not committed.

## Quick start
```bash
source .venv/bin/activate
python -m ipo_rag.cli ingest --docs zomato                       # builds + persists the index for the baseline config
python -m ipo_rag.cli query "What is the offer price?" --docs zomato --no-llm     # retrieval only
export NVIDIA_API_KEY=...                                        # for generation (placeholder backend)
python -m ipo_rag.cli query "..." --docs zomato --set verify=on retrieval_mode=dense
```
`--set` overrides any config field (`chunker=fixed hierarchy=flat table_mode=flat embedder=bge-large ...`). Each distinct ingest config gets its own index directory (`indexes/<hash>/<doc>/`).

## Ablation studies (final set; one factor at a time against the baseline)
| # | Study | Levels |
|---|---|---|
| 1 | Chunking & hierarchy | fixed_flat, section_flat, section_parent (baseline), section_parent_ctx |
| 2 | Parsing & table representation | inline, flat, markdown (baseline); markdown_summary planned |
| 3 | Retrieval mode | dense, bm25, hybrid (baseline) |
| 4 | Numeric & citation verification | off (baseline), on |
| 5 | Embedder | bge-small (baseline), bge-large, e5-large |

Held fixed: RRF weights 1.0/1.0, MiniLM cross-encoder reranker, lost-in-the-middle ordering, parent cap, no query rewriting. Generator chosen by a pre-registered rule on dev (Nemotron nano; bolt used for the cheap ablations); judge: gpt-oss-120b, gold-vs-prediction.

## Headline results (gold_v1; details in `results/SUMMARY.md`)
- **Hybrid retrieval beats dense-only**: pooled dev+test recall@5 0.578 vs 0.468 (p = 0.015). It is the only significant ablation effect; chunking, table mode, embedder size and verification are within noise (about +-0.05).
- **Retrieval is the bottleneck**: giving the generator the gold evidence pages lifts the judge score from 0.59 to 0.84 (dev). Multi-document questions are the weakest group.
- **Verification** cuts unsupported figures (pooled 5.8% -> 1.9%) without changing the judge score.
- **FinanceBench subset** (52 questions, 12 filings; `RAG_CORPUS=financebench RAG_GOLD=fb1`): recall@5 about 0.33; hybrid ties dense; US 10-K layout labels (`RAG_STRUCTURE=sec`) change nothing. The right filing is found 88% of the time; the page is not (look-alike statement tables, computed answers). See `results/report__financebench_analysis.md`.

Reproduce: `python -m ragbench.overnight --lane A` (and B-G; lanes F/G run FinanceBench), `python -m ragbench.summary`, `ragbench/fb_sec.sh`, `python -m ragbench.fb_diagnose`. API keys come from environment variables only (`GROQ_API_KEYS`, `NVIDIA_API_KEYS`; never commit them).

## Design decisions
- **Exact cosine search over persisted numpy embeddings** instead of Qdrant/Chroma: a filing is a few thousand chunks, so brute force is exact (no ANN recall loss to confound the ablations), reproducible and needs no server.
- **Failures are returned, not disguised as answers** (`failed`, `error`), unlike the prototype, so timeouts cannot be scored as answers.
- **Verification "on"** = check every answer number against the retrieved text and every cited page against the retrieved pages; regenerate once with feedback; if figures remain unsupported, abstain.
- Chapter labels come from heading detection (`ipo_rag/structure.py`); they are used for chunk boundaries and sampling, not as ground truth.

## Known limitations / next steps
- Section-aware children average ~790 chars vs ~1000 for fixed windows (study 1 slightly confounded; match sizes or report it).
- `markdown_summary` table level, bge-large / e5-large indexes, and the evaluation adapter for `ipo_rag` are not built yet.
- **Experiments are complete** (dev ablations, generator comparison, verification, hierarchy, embedders, ceiling test, leave-one-out, one-shot test run and test-split confirmations). Read `results/SUMMARY.md`; protocol, rules and bugs found by the evaluation are in `ragbench/README.md`.
- Gold set v1 is built (`eval/gold/gold_v1.json`, 121 items: dev 53 / test 68) but is LLM-generated, filtered and LLM-verified, then treated as audited by the project author (no per-item human decision log is committed). See `goldgen/REVIEW_LOG.md`.
- Citation format: the small placeholder generator often ignores `[Company, Page: N]`.
