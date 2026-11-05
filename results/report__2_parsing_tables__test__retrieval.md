## 2_parsing_tables (test, retrieval, open corpus); baseline = markdown

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
