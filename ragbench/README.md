# ragbench - evaluation stack and experiment runner

Everything is deterministic and resumable: a run is a pure function of (pipeline config, gold-set hash, generator model, judge model); finished
runs are skipped, LLM calls are cached, API keys come only from the environment (`GROQ_API_KEYS`, `NVIDIA_API_KEYS`, comma separated; the pool
uses the first key and fails over on 429/401/403).

## Layers and metrics
| Layer | Metrics | LLM? |
|---|---|---|
| Retrieval (candidates before rerank `cand.*`, final top-5 `final.*`, assembled contexts `ctx.*`) | hit / full-hit / recall / precision @k, MRR, nDCG, scored over **evidence groups** (any one page of a group suffices; every group must be hit) | no |
| Grounding | key-fact recall, numeric support / unsupported figures (page numbers in citations, including ranges, are excluded), citation validity, abstention (correct / false) | no |
| Answer quality (`judge.*`) | gold-vs-prediction judge, 3 axes on a 5-level scale: coverage, faithfulness (accuracy vs gold, none invented), redundancy; overall = mean; abstentions on answerable questions score 0 without a call | gpt-oss-120b (Groq) |
| System | failure rate, latency | no |

Retrieval is scored over the **open corpus** (all 12 filings together, no router); `--oracle` restricts each question to its own filing(s).

## Protocol
* Dev split (5 filings, 53 items) for every ablation and for choosing the generator; test split (7 filings, 68 items) is used once, for the final run.
* Paired statistics: bootstrap CI on per-question differences, exact McNemar with Holm correction within a study.
* **Pre-registered adoption rules** (`ragbench/final.py`): a non-baseline level replaces the baseline only if its improvement is supported (CI above zero
  or Holm p < 0.05 in its favour); verification is adopted only if it cuts unsupported figures without lowering judge.overall by more than 0.03;
  generator = highest mean judge.overall on dev among generators with failure rate <= 10% (ties -> cheaper).
* Study 5 (embedders) uses the dev-filings corpus only, for compute reasons (bge-large/e5-large embed at ~3 chunks/s on this laptop); its baseline row
  is re-evaluated on the same corpus.

## Commands
```bash
python -m ragbench.ablate --study 3_retrieval_mode --split dev                       # retrieval study (no LLM)
python -m ragbench.ablate --study 4_verification --mode generation --gen bolt        # answer-level study (+ judge)
python -m ragbench.ablate --generators bolt nano ultra --split dev                   # generator comparison + selection rule
python -m ragbench.final                                                             # decisions, leave-one-out, one-shot test run, report
python -m ragbench.overnight --lane A|B|C                                            # the overnight queue
python tests/test_ipo_rag.py && python tests/test_ragbench.py                        # model-free tests
```

## Bugs found by the evaluation itself (kept for the record)
1. **Hybrid retrieval was secretly dense.** The inherited 0.6/0.4 RRF weights meant BM25-only hits could never enter the candidate cut-off.
   Fixed with standard equal-weight RRF; guarded by `test_hybrid_actually_mixes_sparse_hits`.
2. **Page ranges counted as financial figures.** `[Page: 266-267]` left `267` behind as an "unsupported figure" (~48% of answers flagged; it also made
   verification abstain on correct answers). Fixed (`cited_pages`, `CITE_SPAN`); guarded by `test_page_ranges_are_not_figures`.
3. **Judge truncated contexts** (earlier context-based judge) and **reasoning models exhausted the token budget** (empty replies); the judge is now
   gold-vs-prediction and generation uses `max_tokens=4096`.
4. A rate-limit bug in gold-set evidence expansion treated unanswered calls as "no"; unchecked pages are now retried.

## Judge quota (Groq free tier) - read before resuming
`openai/gpt-oss-120b` allows **200,000 tokens per day per key** (the window resets at 00:00 UTC, i.e. 05:30 IST), i.e. roughly 150 judge calls per key per day (each call
spends ~1.3k tokens, mostly hidden reasoning). Three keys therefore judge ~450 answers/day. During the first overnight run the quota was exhausted, so:
* all dev experiments and the one-shot test run (E9) are fully judged;
* the *test-split confirmations* of verification / hierarchy and the test ceiling runs were generated but **not yet judged**.
Finish them when quota has recovered (resumable; judged calls are cached):
```bash
export GROQ_API_KEYS=key1,key2,key3 NVIDIA_API_KEYS=...     # never commit keys
python -m ragbench.overnight --lane E        # judge pending runs, regenerate reports, pooled analysis, SUMMARY.md
```
Deterministic metrics (retrieval, grounding, abstention) never need the judge and are complete for every run.
