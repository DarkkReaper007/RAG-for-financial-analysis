# RAG for Indian IPO offer documents (DRHP / RHP / Prospectus)

A question-answering pipeline for SEBI offer documents, with an evaluation framework that tests which design choices actually matter, a small demo app, and the report.

**Main finding.** Hybrid retrieval (keyword plus embeddings) beats embeddings alone (recall@5 0.578 vs 0.468, p = 0.015). Chunking, table format, embedder size and the figure check made no measurable difference. Retrieval, not the generator, limits answer quality: with the right pages the judge score rises from 0.59 to 0.84. Details in [`results/SUMMARY.md`](results/SUMMARY.md) and the report in [`paper/`](paper/).

## Quick start (about 10 minutes): look at the demo

You need Python 3.9 or newer and the 12 filing PDFs.

```bash
git clone https://github.com/DarkkReaper007/RAG-for-financial-analysis.git
cd RAG-for-financial-analysis

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # downloads PyTorch, so it takes a few minutes
```

**Get the PDFs.** They are public SEBI filings but are not stored in git. Ask for the shared folder, put the files in `data/prospectuses/` with the filenames listed in `data/manifest.json`, then check them:

```bash
python scripts/check_data.py         # lists missing or changed files
```

**Start the demo.**

```bash
python -m demo.server
```

Open <http://127.0.0.1:8000>. The six example questions are saved in `demo/cache/`, so they work without any API key and without building indexes. Try the "Table" chip, click the citation to open the source page, then open **Compare** and **Results**.

To ask your own questions the demo needs the search indexes and a generator key (next section).

## Ask your own questions

1. **Build the indexes** (once, about 1.7 GB on disk; the slow part is table detection, so allow an hour or more on a laptop). They are stored in `indexes/` and are not in git. Build a single filing first to try it:
   ```bash
   python -m ipo_rag.cli ingest --docs zomato      # one filing
   python -m ipo_rag.cli ingest                    # all 12
   ```
2. **Set an API key** for the generator, in your shell only. Never write keys into files or commit them. Several keys can be comma separated and are used in turn when one is rate limited:
   ```bash
   export NVIDIA_API_KEYS=key1,key2               # NVIDIA NIM (the generator)
   export GROQ_API_KEYS=key1                      # Groq (only for the answer judge in experiments)
   ```
3. **Try it from the command line, or restart the demo.**
   ```bash
   python -m ipo_rag.cli query "What is the offer price?" --docs zomato --no-llm   # search only
   python -m ipo_rag.cli query "Who are the statutory auditors?" --docs swiggy
   python -m demo.server
   ```
`--set` changes any setting, for example `--set chunker=fixed retrieval_mode=dense verify=on`.
Each distinct ingest setting gets its own index folder (`indexes/<hash>/<filing>/`).

## Review the gold set

```bash
python review_app/app.py                         # http://127.0.0.1:7861
```
Shows each question next to its source pages. Decisions are appended to `eval/gold/review_decisions.jsonl`; `gold_v1.json` is never modified.

## What is in the repository

| Path | What it is |
|---|---|
| `ipo_rag/` | The pipeline: parsing, chunking, indexing, retrieval, generation, verification. Every ablation factor is a field in `config.py` |
| `ragbench/` | Evaluation and experiments: metrics, judge, ablations, statistics, final run. See [`ragbench/README.md`](ragbench/README.md) for the protocol and pre-registered rules |
| `goldgen/` | How the gold set was built (LLM-assisted generation, filters, evidence expansion). See `goldgen/README.md` and `goldgen/REVIEW_LOG.md` |
| `eval/gold/` | `gold_v1.json` (121 questions, 12 filings) and `gold_fb1.json` (a FinanceBench subset used only in side experiments; not part of the report), each with a sha256 |
| `results/` | Report files for every experiment; `SUMMARY.md` is the index, `report__financebench_analysis.md` is a side experiment on FinanceBench that is not part of the report |
| `demo/` | Prospectus Desk web app (FastAPI backend, plain HTML/CSS/JS frontend). See [`demo/README.md`](demo/README.md) |
| `review_app/` | Gradio app for checking the gold set against the PDF pages |
| `paper/` | LaTeX report (`main.tex`), slides (`slides.pptx`) and the scripts that draw the figures |
| `scripts/check_data.py` | Checks that the PDFs are present and unchanged |
| `tests/` | Tests that need no models or network: `python tests/test_ipo_rag.py && python tests/test_ragbench.py` |
| `data/manifest.json` | The 12 filings, the dev/test split by document, and the sha256 of each PDF |

Not in git: PDFs (`data/prospectuses/`, `data/financebench/pdfs/`), built indexes (`indexes/`) and raw per-question results (`ragbench/results/`).

## The five ablation studies

| # | Study | Levels (baseline marked *) |
|---|---|---|
| 1 | Chunking and hierarchy | fixed, section, section + parent*, section + parent + section name |
| 2 | Table representation | inline, flat, markdown* |
| 3 | Retrieval mode | dense, BM25, hybrid* |
| 4 | Figure and citation check | off*, on |
| 5 | Embedder | bge-small*, bge-large, e5-large |

Held fixed: equal-weight rank fusion, MiniLM cross-encoder reranker, context ordering, parent size cap, no query rewriting. The generator (Nemotron nano) was chosen on the dev split by a rule fixed in advance; the judge is gpt-oss-120b comparing the predicted answer with the gold answer.

To re-run experiments see `ragbench/README.md` (`python -m ragbench.ablate ...`, `python -m ragbench.overnight --lane A`, then `python -m ragbench.summary`). Judge calls use the Groq free tier, which allows roughly 150 judged answers per key per day.

## Design decisions

- **Exact cosine search over stored vectors** instead of a vector database. A filing has a few thousand chunks, so brute force is exact (no approximate-search error to confuse the ablations), reproducible, and needs no server.
- **Failures are returned, not hidden.** A timeout or empty reply is marked `failed` and never scored as an answer.
- **Figure check ("verify = on").** Every number in the answer must appear in the retrieved text and every cited page must have been retrieved. The model rewrites once; if figures are still unsupported, the system abstains.
- **Open corpus.** Retrieval searches all 12 filings together, with no document router, so the system also has to pick the right company.
- Chapter labels come from heading detection (`ipo_rag/structure.py`) and are used for chunk boundaries and sampling, not as ground truth. A US 10-K variant is available with `RAG_STRUCTURE=sec` for the FinanceBench runs.

## Known limitations

- The gold set was built with LLM help and audited by the project author; no independent annotators, and the sets are small, so confidence intervals are wide. Null results mean no effect large enough to detect.
- One judge model; checked against a deterministic key-fact metric (84% agreement), not against human grading.
- The report (`paper/main.tex`) and slides have not been compiled or rendered on every machine; check the page count when you build them (`pdflatex main.tex`, twice).

## History

The first prototype (LangChain, Chroma, Ollama, RAGAS) was removed from `main` after the new pipeline replaced it. It is still in the git history, for example at commit `ec2b99d`.
