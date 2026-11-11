# Pooled confirmatory analysis (dev + test)

Decisions were fixed on dev with pre-registered rules; pooling adds statistical power and does not change them. Gold set treated as audited by the authors.

### 3_retrieval_mode (retrieval): pooled n=109 (48 dev + 61 test)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.578 [0.49, 0.67] | 0.473 [0.39, 0.56] | 0.451 [0.37, 0.53] | - | - |
| dense | 0.468 [0.38, 0.56] | 0.365 [0.29, 0.44] | 0.345 [0.27, 0.42] | -0.110 [-0.18, -0.04] | 0.015 (wins 3 / losses 15) |
| bm25 | 0.578 [0.49, 0.67] | 0.462 [0.38, 0.54] | 0.436 [0.36, 0.52] | +0.000 [-0.06, 0.06] | 1.000 (wins 7 / losses 7) |

### 1_chunking_hierarchy (retrieval): pooled n=109 (48 dev + 61 test)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.647 [0.56, 0.73] | 0.540 [0.46, 0.62] | 0.647 [0.56, 0.73] | - | - |
| fixed_flat | 0.661 [0.57, 0.75] | 0.544 [0.46, 0.63] | 0.661 [0.57, 0.75] | +0.014 [-0.05, 0.07] | 1.000 (wins 7 / losses 6) |
| section_flat | 0.647 [0.56, 0.73] | 0.540 [0.46, 0.62] | 0.647 [0.56, 0.73] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.670 [0.58, 0.75] | 0.552 [0.47, 0.63] | 0.670 [0.58, 0.75] | +0.023 [-0.03, 0.08] | 1.000 (wins 7 / losses 4) |

### 2_parsing_tables (retrieval): pooled n=109 (48 dev + 61 test)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.647 [0.56, 0.73] | 0.540 [0.46, 0.62] | 0.647 [0.56, 0.73] | - | - |
| inline | 0.619 [0.53, 0.71] | 0.521 [0.44, 0.60] | 0.628 [0.54, 0.72] | -0.028 [-0.06, 0.00] | 0.500 (wins 0 / losses 3) |
| flat | 0.647 [0.56, 0.73] | 0.542 [0.46, 0.63] | 0.647 [0.56, 0.73] | +0.000 [-0.03, 0.03] | 1.000 (wins 1 / losses 1) |

### 4_verification (generation, nano): pooled n=121 (53 dev + 68 test)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.622 [0.54, 0.71] | 0.058 [0.00, 0.13] | 0.283 [0.20, 0.37] | 0.789 [0.70, 0.87] | 1.000 [1.00, 1.00] | - | - |
| on | 0.636 [0.55, 0.72] | 0.019 [0.00, 0.06] | 0.276 [0.19, 0.36] | 0.793 [0.71, 0.88] | 1.000 [1.00, 1.00] | +0.005 [-0.02, 0.04] | 1.000 (wins 0 / losses 1) |

### 1_chunking_hierarchy (generation, nano): pooled n=121 (53 dev + 68 test)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.622 [0.54, 0.71] | 0.585 [0.49, 0.68] | 0.789 [0.70, 0.87] | 0.569 [0.47, 0.67] | 0.949 [0.88, 1.00] | - | - |
| section_flat | 0.595 [0.51, 0.68] | 0.559 [0.46, 0.65] | 0.812 [0.73, 0.90] | 0.534 [0.44, 0.64] | 1.000 [1.00, 1.00] | -0.021 [-0.07, 0.03] | 0.625 (wins 1 / losses 3) |

### 1_chunking_hierarchy (generation, bolt): pooled n=121 (53 dev + 68 test)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.589 [0.51, 0.67] | 0.585 [0.49, 0.67] | 0.797 [0.71, 0.88] | 0.547 [0.44, 0.65] | 1.000 [1.00, 1.00] | - | - |
| section_flat | 0.496 [0.41, 0.58] | 0.486 [0.39, 0.58] | 0.835 [0.75, 0.91] | 0.494 [0.39, 0.59] | 1.000 [1.00, 1.00] | -0.093 [-0.15, -0.05] | 0.289 (wins 2 / losses 6) |
