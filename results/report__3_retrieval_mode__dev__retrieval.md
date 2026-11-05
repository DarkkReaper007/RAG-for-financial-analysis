## 3_retrieval_mode (dev, retrieval, open corpus); baseline = hybrid

### all items (n=48)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.635 [0.50, 0.77] | 0.488 [0.37, 0.60] | 0.441 [0.33, 0.55] | - | - |
| dense | 0.469 [0.33, 0.60] | 0.371 [0.25, 0.49] | 0.346 [0.23, 0.46] | -0.167 [-0.27, -0.06] | 0.016 (wins 0 / losses 8) |
| bm25 | 0.573 [0.43, 0.71] | 0.458 [0.33, 0.59] | 0.432 [0.31, 0.56] | -0.062 [-0.17, 0.02] | 0.375 (wins 1 / losses 4) |
