## Generator comparison (ext, baseline pipeline config, judge gpt-oss-120b)

| generator | n | failure rate | gen p50 s | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt | 52 | 0% | 1.19 | 0.302 [0.20, 0.42] | 0.255 [0.15, 0.38] | 0.448 [0.28, 0.62] | 0.938 [0.86, 0.99] | 0.278 [0.15, 0.41] | 0.976 [0.94, 1.00] | 0.091 [0.00, 0.23] | 0.942 [0.83, 1.00] | 0.519 [0.38, 0.65] | - |
| nano | 52 | 4% | 4.31 | 0.344 [0.24, 0.45] | 0.255 [0.15, 0.37] | 0.520 [0.34, 0.69] | 1.000 [1.00, 1.00] | 0.149 [0.06, 0.25] | 0.904 [0.76, 1.00] | 0.188 [0.00, 0.38] | 1.000 [1.00, 1.00] | 0.480 [0.34, 0.62] | - |

**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: nano**