## 1_chunking_hierarchy (dev, generation, nano, open corpus); baseline = section_parent

### all items (n=53)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.594 [0.47, 0.72] | 0.560 [0.41, 0.70] | 0.758 [0.61, 0.89] | 0.575 [0.42, 0.72] | 0.960 [0.88, 1.00] | - | - |
| section_flat | 0.578 [0.44, 0.71] | 0.533 [0.39, 0.68] | 0.800 [0.66, 0.93] | 0.538 [0.38, 0.69] | 1.000 [1.00, 1.00] | -0.008 [-0.09, 0.06] | 1.000 (wins 0 / losses 1) |
