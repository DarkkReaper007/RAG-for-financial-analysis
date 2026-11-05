## 4_verification (dev, generation, nano, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=53)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.594 [0.47, 0.72] | 0.040 [0.00, 0.12] | 0.304 [0.17, 0.43] | 0.758 [0.61, 0.89] | 1.000 [1.00, 1.00] | - | - |
| on | 0.606 [0.47, 0.73] | 0.000 [0.00, 0.00] | 0.289 [0.16, 0.42] | 0.750 [0.60, 0.88] | 1.000 [1.00, 1.00] | -0.009 [-0.03, 0.00] | 1.000 (wins 0 / losses 0) |
