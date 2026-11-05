## 1_chunking_hierarchy (test, generation, bolt, open corpus); baseline = section_parent

### all items (n=68)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.597 [0.48, 0.71] | 0.594 [0.48, 0.71] | 0.835 [0.73, 0.92] | 0.573 [0.44, 0.71] | 1.000 [1.00, 1.00] | - | - |
| section_flat | 0.533 [0.42, 0.65] | 0.512 [0.39, 0.64] | 0.838 [0.72, 0.93] | 0.500 [0.36, 0.64] | 1.000 [1.00, 1.00] | -0.064 [-0.13, 0.00] | 0.375 (wins 1 / losses 4) |
