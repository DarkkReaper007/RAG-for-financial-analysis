## 4_verification (dev, generation, bolt, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=53)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.580 [0.45, 0.71] | 0.000 [0.00, 0.00] | 0.312 [0.19, 0.44] | 0.750 [0.61, 0.88] | 0.981 [0.95, 1.00] | - | - |
| on | 0.549 [0.42, 0.68] | 0.000 [0.00, 0.00] | 0.354 [0.23, 0.50] | 0.806 [0.68, 0.92] | 1.000 [1.00, 1.00] | -0.031 [-0.10, 0.03] | 1.000 (wins 0 / losses 0) |
