## Generator comparison (dev, baseline pipeline config, judge gpt-oss-120b)

| generator | n | failure rate | gen p50 s | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt | 53 | 0% | 1.05 | 0.580 [0.45, 0.71] | 0.573 [0.43, 0.71] | 0.750 [0.61, 0.88] | 0.947 [0.89, 0.99] | 0.518 [0.38, 0.66] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.981 [0.95, 1.00] | 0.312 [0.19, 0.44] | 0.400 [0.00, 0.80] |
| nano | 53 | 4% | 4.83 | 0.594 [0.47, 0.72] | 0.560 [0.41, 0.70] | 0.758 [0.61, 0.89] | 1.000 [1.00, 1.00] | 0.575 [0.42, 0.72] | 0.960 [0.88, 1.00] | 0.040 [0.00, 0.12] | 1.000 [1.00, 1.00] | 0.304 [0.17, 0.43] | 0.800 [0.40, 1.00] |
| ultra | 53 | 0% | 8.54 | 0.566 [0.43, 0.70] | 0.578 [0.43, 0.71] | 0.800 [0.68, 0.91] | 0.992 [0.97, 1.00] | 0.563 [0.42, 0.71] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.375 [0.25, 0.52] | 1.000 [1.00, 1.00] |

**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: nano**