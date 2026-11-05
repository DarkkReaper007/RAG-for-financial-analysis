## 2_parsing_tables (ext, retrieval, open corpus); baseline = markdown

### all items (n=52)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | 0.333 [0.21, 0.47] | 0.233 [0.14, 0.33] | 0.372 [0.24, 0.51] | - | - |
| inline | 0.314 [0.19, 0.45] | 0.217 [0.13, 0.32] | 0.353 [0.22, 0.49] | -0.019 [-0.06, 0.00] | 1.000 (wins 0 / losses 1) |
| flat | 0.333 [0.21, 0.47] | 0.226 [0.14, 0.32] | 0.372 [0.24, 0.51] | +0.000 [-0.06, 0.06] | 1.000 (wins 1 / losses 1) |

### numeric + table only (n=0)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **markdown** (baseline) | - | - | - | - | - |
| inline | - | - | - | None | 1.000 (wins 0 / losses 0) |
| flat | - | - | - | None | 1.000 (wins 0 / losses 0) |
