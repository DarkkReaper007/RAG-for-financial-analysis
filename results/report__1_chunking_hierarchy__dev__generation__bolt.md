## 1_chunking_hierarchy (dev, generation, bolt, open corpus); baseline = section_parent

### all items (n=53)

| level | judge.overall | judge.coverage | judge.faithfulness | grounding.key_fact_recall | grounding.numeric_support | Δ judge.overall vs base [95% CI] | McNemar p (grounding.key_fact_recall, Holm) |
|---|---|---|---|---|---|---|---|
| **section_parent** (baseline) | 0.580 [0.45, 0.71] | 0.573 [0.43, 0.71] | 0.750 [0.61, 0.88] | 0.518 [0.38, 0.66] | 1.000 [1.00, 1.00] | - | - |
| section_flat | 0.450 [0.32, 0.59] | 0.453 [0.31, 0.60] | 0.830 [0.69, 0.94] | 0.488 [0.33, 0.64] | 1.000 [1.00, 1.00] | -0.130 [-0.21, -0.06] | 1.000 (wins 1 / losses 2) |
