## 3_retrieval_mode (test, retrieval, open corpus); baseline = hybrid

### all items (n=61)

| level | cand.recall@5 | cand.ndcg@5 | cand.mrr | Δ cand.recall@5 vs base [95% CI] | McNemar p (cand.hit@5, Holm) |
|---|---|---|---|---|---|
| **hybrid** (baseline) | 0.533 [0.41, 0.66] | 0.461 [0.35, 0.58] | 0.459 [0.35, 0.57] | - | - |
| dense | 0.467 [0.34, 0.59] | 0.360 [0.26, 0.47] | 0.344 [0.25, 0.45] | -0.066 [-0.16, 0.03] | 0.688 (wins 3 / losses 7) |
| bm25 | 0.582 [0.46, 0.70] | 0.465 [0.36, 0.57] | 0.438 [0.33, 0.55] | +0.049 [-0.04, 0.14] | 0.688 (wins 6 / losses 3) |
