# Gold-set generation (LLM-assisted, human-verified)

Built from raw PyMuPDF page text, independent of the pipeline's parser/chunker (only chapter labels are shared, as topic tags).

| Stage | Script | Status |
|---|---|---|
| 0 Extract pages (+ chapter labels, table-like flag) | `extract_pages.py` | built, run on all 12 filings |
| 1 Stratified page sample (chapter round-robin, 25% table pages) | `sample_pages.py` | built |
| 2 Question generation (hosted LLM, JSON: question/answer/quote) | `gen_questions.py` | built |
| 3 Automatic filters (quote on page, no placeholder answer, self-contained, company named, no 6-gram copy, near-duplicate, independent LLM verifier) | `filter_candidates.py` | built |
| 4 Evidence expansion (pool other pages that answer the question) | `expand_evidence.py` | built, run |
| 4b Rating + quota selection + reviewer exclusions | `rate_select.py`, `review_exclusions.json` | built, run |
| 5 Multi-document + unanswerable items | `gen_special.py` | built, run (12 + 12) |
| 6 Human review UI | - | **not built** (review done by Claude, see `REVIEW_LOG.md`) |
| 7-8 Assemble, validate, hash | `build_gold.py`, `validate_gold.py` | built, run: `eval/gold/gold_v1.json`, 121 items, 0 problems |

Status: **gold_v1 exists (121 items, LLM-verified + Claude spot-check; verified by all four project authors).**

```bash
export NVIDIA_API_KEY=...        # never commit; read from the environment only
python goldgen/extract_pages.py
python goldgen/sample_pages.py --docs zomato ola_electric --per-doc 20
python goldgen/gen_questions.py                     # generator: nvidia/nemotron-3.5-bolt-e4b
python goldgen/filter_candidates.py                 # verifier:  nvidia/nemotron-3.5-lightning-30b-a3b (reasoning model)
python goldgen/rate_select.py && python goldgen/expand_evidence.py && python goldgen/gen_special.py
python goldgen/build_gold.py && python goldgen/validate_gold.py
# GROQ_API_KEY (optional) is used as the fallback judge when NVIDIA rate-limits
```
Outputs go to `goldgen/out/` (ignored). All model calls are cached; empty replies are never cached.

## Pilot (zomato prospectus + ola_electric DRHP, 20 pages each)
40 sampled pages -> 39 candidates -> 24 pass the automatic filters (15 fail: 8 verifier-rejected, 4 copied 6-grams, 2 verifier no reply, 1 quote not on page).
Reading the questions by eye: grounded and correct, but many are narrow detail lookups with the company name appended awkwardly; the human review step must cull these, and investor-style, multi-section, multi-document and unanswerable questions need to be written by hand.

## Corpus notes
- DRHPs leave the price as `[●]`, so price-band questions on a DRHP have no answer (the generator is told to skip placeholders).
- Split by document (5 dev / 7 test) is fixed in `data/manifest.json`.
