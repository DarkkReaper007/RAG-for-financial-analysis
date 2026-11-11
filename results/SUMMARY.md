# Experiment summary (auto-generated)

All results use the LLM-built gold set audited by the authors (gold_v1). Dev = model/level selection; test = one-shot final run + confirmations.

## Final configuration (pre-registered rules applied on dev)

* Overrides vs baseline: `none: baseline is the final configuration`
* Generator: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

## E1 Retrieval mode (study 3), dev

### 3_retrieval_mode (dev, retrieval, open corpus); baseline = hybrid

### all items (n=48)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.635 [0.50, 0.77] | 0.488 [0.37, 0.60] | 0.441 [0.33, 0.55] | - | - |
| dense | 0.469 [0.33, 0.60] | 0.371 [0.25, 0.49] | 0.346 [0.23, 0.46] | -0.167 [-0.27, -0.06] | 0.016 (wins 0 / losses 8) |
| bm25 | 0.573 [0.43, 0.71] | 0.458 [0.33, 0.59] | 0.432 [0.31, 0.56] | -0.062 [-0.17, 0.02] | 0.375 (wins 1 / losses 4) |

## E1' Retrieval mode, TEST confirmation

### 3_retrieval_mode (test, retrieval, open corpus); baseline = hybrid

### all items (n=61)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.533 [0.41, 0.66] | 0.461 [0.35, 0.58] | 0.459 [0.35, 0.57] | - | - |
| dense | 0.467 [0.34, 0.59] | 0.360 [0.26, 0.47] | 0.344 [0.25, 0.45] | -0.066 [-0.16, 0.03] | 0.688 (wins 3 / losses 7) |
| bm25 | 0.582 [0.46, 0.70] | 0.465 [0.36, 0.57] | 0.438 [0.33, 0.55] | +0.049 [-0.04, 0.14] | 0.688 (wins 6 / losses 3) |

## E2 Chunking & hierarchy, retrieval (study 1), dev

### 1_chunking_hierarchy (dev, retrieval, open corpus); baseline = section_parent

### all items (n=48)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.635 [0.50, 0.77] | 0.547 [0.42, 0.67] | 0.635 [0.50, 0.77] | - | - |
| fixed_flat | 0.625 [0.48, 0.76] | 0.528 [0.40, 0.65] | 0.625 [0.48, 0.76] | -0.010 [-0.08, 0.06] | 1.000 (wins 2 / losses 2) |
| section_flat | 0.635 [0.50, 0.77] | 0.547 [0.42, 0.67] | 0.635 [0.50, 0.77] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.625 [0.48, 0.76] | 0.516 [0.39, 0.64] | 0.625 [0.48, 0.76] | -0.010 [-0.10, 0.08] | 1.000 (wins 3 / losses 3) |

## E2' Chunking & hierarchy, retrieval, TEST confirmation

### 1_chunking_hierarchy (test, retrieval, open corpus); baseline = section_parent

### all items (n=61)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.656 [0.54, 0.77] | - | - |
| fixed_flat | 0.689 [0.57, 0.80] | 0.557 [0.45, 0.66] | 0.689 [0.57, 0.80] | +0.033 [-0.05, 0.12] | 1.000 (wins 5 / losses 4) |
| section_flat | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.656 [0.54, 0.77] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.705 [0.59, 0.81] | 0.580 [0.48, 0.68] | 0.705 [0.59, 0.81] | +0.049 [0.00, 0.11] | 1.000 (wins 4 / losses 1) |

## E3 Parsing & table representation (study 2), dev

### 2_parsing_tables (dev, retrieval, open corpus); baseline = markdown

### all items (n=48)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.635 [0.50, 0.77] | 0.547 [0.42, 0.67] | 0.635 [0.50, 0.77] | - | - |
| inline | 0.573 [0.43, 0.71] | 0.510 [0.37, 0.64] | 0.594 [0.45, 0.73] | -0.062 [-0.15, 0.00] | 0.500 (wins 0 / losses 3) |
| flat | 0.656 [0.51, 0.79] | 0.568 [0.44, 0.69] | 0.656 [0.51, 0.79] | +0.021 [0.00, 0.06] | 1.000 (wins 1 / losses 0) |

### numeric + table only (n=20)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.800 [0.60, 0.95] | 0.710 [0.53, 0.88] | 0.800 [0.60, 0.95] | - | - |
| inline | 0.700 [0.50, 0.90] | 0.638 [0.44, 0.83] | 0.700 [0.50, 0.90] | -0.100 [-0.25, 0.00] | 1.000 (wins 0 / losses 2) |
| flat | 0.850 [0.70, 1.00] | 0.760 [0.59, 0.91] | 0.850 [0.70, 1.00] | +0.050 [0.00, 0.15] | 1.000 (wins 1 / losses 0) |

## E3' Parsing & table representation, TEST confirmation

### 2_parsing_tables (test, retrieval, open corpus); baseline = markdown

### all items (n=61)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.656 [0.54, 0.77] | - | - |
| inline | 0.656 [0.54, 0.77] | 0.530 [0.43, 0.64] | 0.656 [0.54, 0.77] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| flat | 0.639 [0.52, 0.75] | 0.522 [0.42, 0.63] | 0.639 [0.52, 0.75] | -0.016 [-0.05, 0.00] | 1.000 (wins 0 / losses 1) |

### numeric + table only (n=27)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.741 [0.56, 0.89] | 0.604 [0.45, 0.76] | 0.741 [0.56, 0.89] | - | - |
| inline | 0.741 [0.56, 0.89] | 0.595 [0.44, 0.75] | 0.741 [0.56, 0.89] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| flat | 0.704 [0.52, 0.85] | 0.584 [0.42, 0.74] | 0.704 [0.52, 0.85] | -0.037 [-0.11, 0.00] | 1.000 (wins 0 / losses 1) |

## E4 Embedder (study 5), dev filings corpus

### 5_embedder (dev, retrieval, open corpus); baseline = bge-small

### all items (n=48)

| level | final.recall@5 | final.ndcg@5 | cand.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **bge-small** (baseline) | 0.698 [0.56, 0.82] | 0.603 [0.48, 0.72] | 0.740 [0.61, 0.85] | - | - |
| bge-large | 0.719 [0.58, 0.83] | 0.614 [0.49, 0.73] | 0.760 [0.64, 0.88] | +0.021 [0.00, 0.06] | 1.000 (wins 1 / losses 0) |
| e5-large | 0.698 [0.56, 0.82] | 0.587 [0.46, 0.71] | 0.740 [0.60, 0.85] | +0.000 [-0.06, 0.06] | 1.000 (wins 1 / losses 1) |

## E4' Embedder, TEST filings corpus (confirmation)

### 5_embedder (test, retrieval, open corpus); baseline = bge-small

### all items (n=61)

| level | final.recall@5 | final.ndcg@5 | cand.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **bge-small** (baseline) | 0.672 [0.55, 0.79] | 0.541 [0.44, 0.65] | 0.689 [0.57, 0.80] | - | - |
| bge-large | 0.680 [0.57, 0.80] | 0.536 [0.44, 0.64] | 0.746 [0.63, 0.85] | +0.008 [-0.05, 0.07] | 1.000 (wins 2 / losses 2) |
| e5-large | 0.705 [0.59, 0.82] | 0.543 [0.44, 0.64] | 0.730 [0.61, 0.84] | +0.033 [0.00, 0.08] | 1.000 (wins 2 / losses 0) |

## E5 Verification, answer level, nano, dev

### 4_verification (dev, generation, nano, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=53)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.594 [0.47, 0.72] | 0.040 [0.00, 0.12] | 0.304 [0.17, 0.43] | 0.758 [0.61, 0.89] | 1.000 [1.00, 1.00] | - | - |
| on | 0.606 [0.47, 0.73] | 0.000 [0.00, 0.00] | 0.289 [0.16, 0.42] | 0.750 [0.60, 0.88] | 1.000 [1.00, 1.00] | -0.009 [-0.03, 0.00] | 1.000 (wins 0 / losses 0) |

## E5 Verification, answer level, bolt, dev

### 4_verification (dev, generation, bolt, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=53)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.580 [0.45, 0.71] | 0.000 [0.00, 0.00] | 0.312 [0.19, 0.44] | 0.750 [0.61, 0.88] | 0.981 [0.95, 1.00] | - | - |
| on | 0.549 [0.42, 0.68] | 0.000 [0.00, 0.00] | 0.354 [0.23, 0.50] | 0.806 [0.68, 0.92] | 1.000 [1.00, 1.00] | -0.031 [-0.10, 0.03] | 1.000 (wins 0 / losses 0) |

## E5' Verification, nano, TEST confirmation

### 4_verification (test, generation, nano, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=68)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.643 [0.53, 0.75] | 0.074 [0.00, 0.19] | 0.267 [0.17, 0.38] | 0.812 [0.70, 0.91] | 1.000 [1.00, 1.00] | - | - |
| on | 0.658 [0.55, 0.76] | 0.034 [0.00, 0.10] | 0.267 [0.17, 0.38] | 0.824 [0.72, 0.92] | 1.000 [1.00, 1.00] | +0.015 [-0.03, 0.07] | 1.000 (wins 0 / losses 1) |

## E6 Hierarchy (parent vs flat), answer level, nano, dev

### 1_chunking_hierarchy (dev, generation, nano, open corpus); baseline = section_parent

### all items (n=53)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.594 [0.47, 0.72] | 0.560 [0.41, 0.70] | 0.758 [0.61, 0.89] | 0.575 [0.42, 0.72] | 0.960 [0.88, 1.00] | - | - |
| section_flat | 0.578 [0.44, 0.71] | 0.533 [0.39, 0.68] | 0.800 [0.66, 0.93] | 0.538 [0.38, 0.69] | 1.000 [1.00, 1.00] | -0.008 [-0.09, 0.06] | 1.000 (wins 0 / losses 1) |

## E6 Hierarchy, answer level, bolt, dev

### 1_chunking_hierarchy (dev, generation, bolt, open corpus); baseline = section_parent

### all items (n=53)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.580 [0.45, 0.71] | 0.573 [0.43, 0.71] | 0.750 [0.61, 0.88] | 0.518 [0.38, 0.66] | 1.000 [1.00, 1.00] | - | - |
| section_flat | 0.450 [0.32, 0.59] | 0.453 [0.31, 0.60] | 0.830 [0.69, 0.94] | 0.488 [0.33, 0.64] | 1.000 [1.00, 1.00] | -0.130 [-0.21, -0.06] | 1.000 (wins 1 / losses 2) |

## E6' Hierarchy, nano, TEST confirmation

### 1_chunking_hierarchy (test, generation, nano, open corpus); baseline = section_parent

### all items (n=68)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.643 [0.53, 0.75] | 0.604 [0.48, 0.72] | 0.812 [0.70, 0.91] | 0.564 [0.43, 0.70] | 0.938 [0.84, 1.00] | - | - |
| section_flat | 0.608 [0.50, 0.72] | 0.578 [0.46, 0.70] | 0.821 [0.70, 0.92] | 0.531 [0.40, 0.67] | 1.000 [1.00, 1.00] | -0.031 [-0.09, 0.03] | 1.000 (wins 1 / losses 2) |

## E6' Hierarchy, bolt, TEST confirmation

### 1_chunking_hierarchy (test, generation, bolt, open corpus); baseline = section_parent

### all items (n=68)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.597 [0.48, 0.71] | 0.594 [0.48, 0.71] | 0.835 [0.73, 0.92] | 0.573 [0.44, 0.71] | 1.000 [1.00, 1.00] | - | - |
| section_flat | 0.533 [0.42, 0.65] | 0.512 [0.39, 0.64] | 0.838 [0.72, 0.93] | 0.500 [0.36, 0.64] | 1.000 [1.00, 1.00] | -0.064 [-0.13, 0.00] | 0.375 (wins 1 / losses 4) |

## E7 Generator comparison, dev (selection rule applied here)

### Generator comparison (dev, baseline pipeline config, judge gpt-oss-120b)

| generator | n | failure rate | gen p50 s | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt | 53 | 0% | 1.05 | 0.580 [0.45, 0.71] | 0.573 [0.43, 0.71] | 0.750 [0.61, 0.88] | 0.947 [0.89, 0.99] | 0.518 [0.38, 0.66] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.981 [0.95, 1.00] | 0.312 [0.19, 0.44] | 0.400 [0.00, 0.80] |
| nano | 53 | 4% | 4.83 | 0.594 [0.47, 0.72] | 0.560 [0.41, 0.70] | 0.758 [0.61, 0.89] | 1.000 [1.00, 1.00] | 0.575 [0.42, 0.72] | 0.960 [0.88, 1.00] | 0.040 [0.00, 0.12] | 1.000 [1.00, 1.00] | 0.304 [0.17, 0.43] | 0.800 [0.40, 1.00] |
| ultra | 53 | 0% | 8.54 | 0.566 [0.43, 0.70] | 0.578 [0.43, 0.71] | 0.800 [0.68, 0.91] | 0.992 [0.97, 1.00] | 0.563 [0.42, 0.71] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.375 [0.25, 0.52] | 1.000 [1.00, 1.00] |

**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: nano**

## E7' Generator comparison, TEST confirmation

### Generator comparison (test, baseline pipeline config, judge gpt-oss-120b)

| generator | n | failure rate | gen p50 s | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt | 68 | 0% | 1.87 | 0.597 [0.48, 0.71] | 0.594 [0.48, 0.71] | 0.835 [0.73, 0.92] | 0.945 [0.90, 0.98] | 0.573 [0.44, 0.71] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.990 [0.97, 1.00] | 0.328 [0.21, 0.44] | 0.571 [0.14, 0.86] |
| nano | 68 | 1% | 3.36 | 0.643 [0.53, 0.75] | 0.604 [0.48, 0.72] | 0.812 [0.70, 0.91] | 0.994 [0.98, 1.00] | 0.564 [0.43, 0.70] | 0.938 [0.84, 1.00] | 0.074 [0.00, 0.19] | 1.000 [1.00, 1.00] | 0.267 [0.17, 0.38] | 1.000 [1.00, 1.00] |
| ultra | 68 | 0% | 6.48 | 0.650 [0.54, 0.76] | 0.643 [0.52, 0.76] | 0.884 [0.80, 0.95] | 0.971 [0.94, 0.99] | 0.667 [0.52, 0.79] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.295 [0.18, 0.41] | 1.000 [1.00, 1.00] |

**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: ultra**

## Ceiling vs end-to-end (dev)

### Ceiling (generator given the GOLD evidence pages) vs end-to-end (baseline retrieval), dev, answerable items

judge.overall: failed generations are excluded from each mean; the paired difference uses items answered in both runs.

| generator | group | n | end-to-end | ceiling (perfect retrieval) | retrieval loss (ceiling - end-to-end) [95% CI] |
|---|---|---|---|---|---|
| nano | ALL | 48 | 0.59 | 0.84 | +0.27 [+0.10, +0.43] |
| nano | definition | 3 | 0.64 | 0.92 | +0.28 |
| nano | factoid | 10 | 0.53 | 0.83 | +0.30 [-0.17, +0.70] |
| nano | list | 10 | 0.75 | 0.72 | +0.06 [-0.27, +0.38] |
| nano | multi_document | 5 | 0.20 | 0.60 | +0.40 [+0.00, +0.80] |
| nano | numeric | 10 | 0.56 | 0.93 | +0.37 [+0.11, +0.67] |
| nano | table | 10 | 0.73 | 1.00 | +0.25 [+0.00, +0.62] |
| bolt | ALL | 48 | 0.58 | 0.79 | +0.21 [+0.04, +0.38] |
| bolt | definition | 3 | 0.33 | 0.97 | +0.64 |
| bolt | factoid | 10 | 0.42 | 0.82 | +0.40 [-0.07, +0.80] |
| bolt | list | 10 | 0.71 | 0.72 | +0.01 [-0.28, +0.29] |
| bolt | multi_document | 5 | 0.00 | 0.75 | +0.75 [+0.35, +1.00] |
| bolt | numeric | 10 | 0.76 | 0.73 | -0.02 [-0.39, +0.33] |
| bolt | table | 10 | 0.79 | 0.87 | +0.08 [-0.17, +0.37] |

## E8 Leave-one-out, dev

### Leave-one-out (dev, open corpus): final configuration vs final with one component replaced

| component replaced | final.recall@5 | Δ vs final [95% CI] | McNemar p (final.hit@5) |
|---|---|---|---|
| (none: final config) | 0.635 [0.50, 0.77] | - | - |
| retrieval_mode -> dense | 0.469 [0.33, 0.60] | -0.167 [-0.27, -0.06] | 0.008 |
| chunker -> fixed | 0.625 [0.48, 0.76] | -0.010 [-0.08, 0.06] | 1.000 |
| table_mode -> inline | 0.573 [0.43, 0.71] | -0.062 [-0.15, 0.00] | 0.250 |

## E9/E10 Final test-split run


Generator: **nano** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`); judge: gpt-oss-120b (gold-vs-prediction). Final-config overrides vs baseline: `none (baseline)`

## final (n=68, failure rate 1%)

| group | n | recall@5 | ndcg@5 | full_hit@5 | recall@10 | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 68 | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.639 [0.52, 0.75] | 0.656 [0.54, 0.77] | 0.651 [0.54, 0.76] | 0.621 [0.50, 0.74] | 0.824 [0.72, 0.91] | 0.994 [0.98, 1.00] | 0.580 [0.44, 0.71] | 0.938 [0.84, 1.00] | 0.074 [0.00, 0.19] | 1.000 [1.00, 1.00] | 0.267 [0.17, 0.38] | 1.000 [1.00, 1.00] |
| definition | 3 | 0.333 | 0.333 | 0.333 | 0.333 | 0.639 | 0.667 | 0.875 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.333 | - |
| factoid | 13 | 0.769 [0.54, 1.00] | 0.630 [0.41, 0.84] | 0.769 [0.54, 1.00] | 0.769 [0.54, 1.00] | 0.590 [0.33, 0.82] | 0.538 [0.23, 0.77] | 0.778 [0.44, 1.00] | 1.000 [1.00, 1.00] | 0.455 [0.18, 0.73] | 0.750 | 0.250 | 1.000 [1.00, 1.00] | 0.308 [0.08, 0.54] | - |
| list | 11 | 0.727 [0.45, 1.00] | 0.553 [0.31, 0.78] | 0.727 [0.45, 1.00] | 0.727 [0.45, 1.00] | 0.758 [0.51, 0.94] | 0.750 [0.50, 0.95] | 0.889 [0.78, 0.97] | 0.972 [0.92, 1.00] | 0.667 | 0.667 | 0.500 | 1.000 [1.00, 1.00] | 0.182 [0.00, 0.45] | - |
| multi_document | 7 | 0.143 [0.00, 0.36] | 0.143 [0.00, 0.34] | 0.000 [0.00, 0.00] | 0.143 [0.00, 0.36] | 0.286 [0.00, 0.71] | 0.286 [0.00, 0.71] | 1.000 | 1.000 | 0.250 [0.00, 0.58] | - | - | 1.000 | 0.714 [0.29, 1.00] | - |
| numeric | 13 | 0.462 [0.23, 0.69] | 0.405 [0.17, 0.66] | 0.462 [0.23, 0.69] | 0.462 [0.23, 0.69] | 0.556 [0.31, 0.81] | 0.500 [0.25, 0.75] | 0.750 [0.38, 1.00] | 1.000 [1.00, 1.00] | 0.500 [0.25, 0.75] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.333 [0.08, 0.58] | - |
| table | 14 | 1.000 [1.00, 1.00] | 0.790 [0.66, 0.91] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.893 [0.76, 0.99] | 0.857 [0.64, 1.00] | 0.821 [0.62, 0.98] | 1.000 [1.00, 1.00] | 0.839 [0.64, 1.00] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | - |
| unanswerable | 7 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1.000 [1.00, 1.00] |

Failure taxonomy: good answer: 34; abstained, evidence not retrieved: 14; unanswerable: correctly abstained: 7; answered, evidence not retrieved: 6; evidence retrieved but answer poor: 4; abstained although evidence exists: 2; generation failed: 1

Lowest-scoring answerable items: t-fac-02 (factoid, judge 0.0), t-num-02 (numeric, judge 0.0), t-lis-01 (list, judge 0.0), t-lis-02 (list, judge 0.0), t-def-01 (definition, judge 0.0), t-fac-04 (factoid, judge 0.0), t-num-05 (numeric, judge 0.0), t-fac-06 (factoid, judge 0.0)

## Oracle retrieval (each question restricted to its own filing(s)), final configuration

| metric | value |
|---|---|
| final.recall@5 | 0.787 [0.68, 0.89] |
| final.ndcg@5 | 0.655 [0.56, 0.75] |
| final.full_hit@5 | 0.770 [0.66, 0.87] |
| cand.recall@10 | 0.803 [0.70, 0.89] |

## Judge vs deterministic key-fact check (dev)

### Judge vs deterministic key-fact recall (dev, answerable, non-abstained, non-failed answers with key facts)

n = 83 answers pooled over bolt/nano/ultra; Pearson r(judge.coverage, key_fact_recall) = 0.56

| | all key facts present | some key fact missing |
|---|---|---|
| judge coverage >= 0.75 | 60 | 9 |
| judge coverage < 0.75 | 4 | 10 |

Agreement between the two independent checks: 84%

Disagreements by type: {'numeric': 3, 'table': 2, 'definition': 3, 'list': 4, 'factoid': 1}
