## 1_chunking_hierarchy (test, generation, nano, open corpus); baseline = section_parent

### all items (n=68)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.643 [0.53, 0.75] | 0.604 [0.48, 0.72] | 0.812 [0.70, 0.91] | 0.564 [0.43, 0.70] | 0.938 [0.84, 1.00] | - | - |
| section_flat | 0.608 [0.50, 0.72] | 0.578 [0.46, 0.70] | 0.821 [0.70, 0.92] | 0.531 [0.40, 0.67] | 1.000 [1.00, 1.00] | -0.031 [-0.09, 0.03] | 1.000 (wins 1 / losses 2) |
