## 1_chunking_hierarchy (dev, retrieval, open corpus); baseline = section_parent

### all items (n=48)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.635 [0.50, 0.77] | 0.547 [0.42, 0.67] | 0.635 [0.50, 0.77] | - | - |
| fixed_flat | 0.625 [0.48, 0.76] | 0.528 [0.40, 0.65] | 0.625 [0.48, 0.76] | -0.010 [-0.08, 0.06] | 1.000 (wins 2 / losses 2) |
| section_flat | 0.635 [0.50, 0.77] | 0.547 [0.42, 0.67] | 0.635 [0.50, 0.77] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.625 [0.48, 0.76] | 0.516 [0.39, 0.64] | 0.625 [0.48, 0.76] | -0.010 [-0.10, 0.08] | 1.000 (wins 3 / losses 3) |
