## 1_chunking_hierarchy (test, retrieval, open corpus); baseline = section_parent

### all items (n=61)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.656 [0.54, 0.77] | - | - |
| fixed_flat | 0.689 [0.57, 0.80] | 0.557 [0.45, 0.66] | 0.689 [0.57, 0.80] | +0.033 [-0.05, 0.12] | 1.000 (wins 5 / losses 4) |
| section_flat | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.656 [0.54, 0.77] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.705 [0.59, 0.81] | 0.580 [0.48, 0.68] | 0.705 [0.59, 0.81] | +0.049 [0.00, 0.11] | 1.000 (wins 4 / losses 1) |
