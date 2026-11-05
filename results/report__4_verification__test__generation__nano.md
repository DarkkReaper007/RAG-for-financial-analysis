## 4_verification (test, generation, nano, open corpus); baseline = off

*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*

### all items (n=68)

| level | judge.overall | grounding.unsupported_figure | grounding.false_abstention | judge.faithfulness | grounding.citation_valid | Δ judge.overall vs base [95% CI] | McNemar p (grounding.unsupported_figure, Holm) |
|---|---|---|---|---|---|---|---|
| **off** (baseline) | 0.643 [0.53, 0.75] | 0.074 [0.00, 0.19] | 0.267 [0.17, 0.38] | 0.812 [0.70, 0.91] | 1.000 [1.00, 1.00] | - | - |
| on | 0.658 [0.55, 0.76] | 0.034 [0.00, 0.10] | 0.267 [0.17, 0.38] | 0.824 [0.72, 0.92] | 1.000 [1.00, 1.00] | +0.015 [-0.03, 0.07] | 1.000 (wins 0 / losses 1) |
