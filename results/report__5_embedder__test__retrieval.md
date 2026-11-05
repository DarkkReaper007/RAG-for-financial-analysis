## 5_embedder (test, retrieval, open corpus); baseline = bge-small

### all items (n=61)

| level | final.recall@5 | final.ndcg@5 | cand.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **bge-small** (baseline) | 0.672 [0.55, 0.79] | 0.541 [0.44, 0.65] | 0.689 [0.57, 0.80] | - | - |
| bge-large | 0.680 [0.57, 0.80] | 0.536 [0.44, 0.64] | 0.746 [0.63, 0.85] | +0.008 [-0.05, 0.07] | 1.000 (wins 2 / losses 2) |
| e5-large | 0.705 [0.59, 0.82] | 0.543 [0.44, 0.64] | 0.730 [0.61, 0.84] | +0.033 [0.00, 0.08] | 1.000 (wins 2 / losses 0) |
