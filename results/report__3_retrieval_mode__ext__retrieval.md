## 3_retrieval_mode (ext, retrieval, open corpus); baseline = hybrid

### all items (n=52)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.276 [0.16, 0.40] | 0.204 [0.11, 0.30] | 0.204 [0.11, 0.31] | - | - |
| dense | 0.276 [0.16, 0.40] | 0.243 [0.14, 0.36] | 0.257 [0.15, 0.37] | +0.000 [-0.10, 0.10] | 1.000 (wins 3 / losses 3) |
| bm25 | 0.122 [0.04, 0.22] | 0.108 [0.04, 0.20] | 0.114 [0.04, 0.20] | -0.154 [-0.27, -0.04] | 0.043 (wins 1 / losses 9) |
