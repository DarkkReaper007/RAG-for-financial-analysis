## 5_embedder (dev, retrieval, open corpus); baseline = bge-small

### all items (n=48)

| level | final.recall@5 | final.ndcg@5 | cand.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **bge-small** (baseline) | 0.698 [0.56, 0.82] | 0.603 [0.48, 0.72] | 0.740 [0.61, 0.85] | - | - |
| bge-large | 0.719 [0.58, 0.83] | 0.614 [0.49, 0.73] | 0.760 [0.64, 0.88] | +0.021 [0.00, 0.06] | 1.000 (wins 1 / losses 0) |
| e5-large | 0.698 [0.56, 0.82] | 0.587 [0.46, 0.71] | 0.740 [0.60, 0.85] | +0.000 [-0.06, 0.06] | 1.000 (wins 1 / losses 1) |
