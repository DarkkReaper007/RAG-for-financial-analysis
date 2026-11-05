## Leave-one-out (dev, open corpus): final configuration vs final with one component replaced

| component replaced | final.recall@5 | Δ vs final [95% CI] | McNemar p (final.hit@5) |
|---|---|---|---|
| (none: final config) | 0.635 [0.50, 0.77] | - | - |
| retrieval_mode -> dense | 0.469 [0.33, 0.60] | -0.167 [-0.27, -0.06] | 0.008 |
| chunker -> fixed | 0.625 [0.48, 0.76] | -0.010 [-0.08, 0.06] | 1.000 |
| table_mode -> inline | 0.573 [0.43, 0.71] | -0.062 [-0.15, 0.00] | 0.250 |