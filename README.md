# RAG for Indian IPO offer documents (DRHP / RHP / Prospectus)

Config-driven RAG pipeline + gold-set generation + evaluation suite, built to support an ablation study.
Work branch: `final`. The older prototype lives in `src/` and `app.py` (kept for reference, not used by the new code).

## Layout
| Path | Purpose |
|---|---|
| `ipo_rag/` | **New pipeline.** `config.py` (every ablation factor), `parsing`, `chunking`, `indexing` (persisted, exact cosine), `retrieval`, `generation`, `verification`, `pipeline`, `cli` |
| `goldgen/` | LLM-assisted gold-set generation (complete; see `goldgen/README.md`) |
| `eval/gold/` | `gold_v1.json` + sha256 |
| `eval/` | Evaluation suite scaffold (metrics, runner, judge, comparer); to be adapted to `ipo_rag` |
| `data/manifest.json` | 12 filings, fixed dev (5) / test (7) split by document, sha256 of each PDF |
| `tests/` | Model-free unit tests: `python tests/test_ipo_rag.py` |

PDFs (`data/prospectuses/`) and built indexes (`indexes/`) are not committed.

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

Held fixed: RRF weights 0.6/0.4, MiniLM cross-encoder reranker, lost-in-the-middle ordering, parent cap, no query rewriting. The generator is undecided (placeholder: a hosted Nemotron model).

## Design decisions
- **Exact cosine search over persisted numpy embeddings** instead of Qdrant/Chroma: a filing is a few thousand chunks, so brute force is exact (no ANN recall loss to confound the ablations), reproducible and needs no server.
- **Failures are returned, not disguised as answers** (`failed`, `error`), unlike the prototype, so timeouts cannot be scored as answers.
- **Verification "on"** = check every answer number against the retrieved text and every cited page against the retrieved pages; regenerate once with feedback; if figures remain unsupported, abstain.
- Chapter labels come from heading detection (`ipo_rag/structure.py`); they are used for chunk boundaries and sampling, not as ground truth.

## Known limitations / next steps
- Section-aware children average ~790 chars vs ~1000 for fixed windows (study 1 slightly confounded; match sizes or report it).
- `markdown_summary` table level, bge-large / e5-large indexes, and the evaluation adapter for `ipo_rag` are not built yet.
- Gold set v1 is built (`eval/gold/gold_v1.json`, 121 items: dev 53 / test 68) but is LLM-verified + Claude spot-check, not human-verified. See `goldgen/REVIEW_LOG.md`.
- Citation format: the small placeholder generator often ignores `[Company, Page: N]`.
