## 4_verification (ext, generation, nano, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=52)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.344 [0.24, 0.45] | 0.188 [0.00, 0.38] | 0.480 [0.34, 0.62] | 0.520 [0.34, 0.69] | 1.000 [1.00, 1.00] | - | - |
| on | 0.310 [0.20, 0.43] | 0.278 [0.11, 0.50] | 0.580 [0.44, 0.72] | 0.607 [0.42, 0.80] | 1.000 [1.00, 1.00] | -0.035 [-0.10, 0.02] | 1.000 (wins 1 / losses 0) |
