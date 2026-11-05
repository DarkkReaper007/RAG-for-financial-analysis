## 2_parsing_tables (dev, retrieval, open corpus); baseline = markdown

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
