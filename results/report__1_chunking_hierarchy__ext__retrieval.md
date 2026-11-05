## 1_chunking_hierarchy (ext, retrieval, open corpus); baseline = section_parent

### all items (n=52)

| level | final.recall@5 | final.ndcg@5 | ctx.recall@10 | Δ final.recall@5 vs base [95% CI] | McNemar p (final.hit@5, Holm) |
|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.333 [0.21, 0.47] | 0.233 [0.14, 0.33] | 0.372 [0.24, 0.51] | - | - |
| fixed_flat | 0.304 [0.18, 0.43] | 0.238 [0.14, 0.34] | 0.304 [0.18, 0.43] | -0.029 [-0.12, 0.06] | 1.000 (wins 3 / losses 4) |
| section_flat | 0.333 [0.21, 0.47] | 0.233 [0.14, 0.33] | 0.333 [0.21, 0.47] | +0.000 [0.00, 0.00] | 1.000 (wins 0 / losses 0) |
| section_parent_ctx | 0.269 [0.15, 0.38] | 0.196 [0.11, 0.29] | 0.321 [0.20, 0.45] | -0.064 [-0.18, 0.04] | 1.000 (wins 3 / losses 7) |
