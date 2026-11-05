## Generator comparison (test, baseline pipeline config, judge gpt-oss-120b)

| generator | n | failure rate | gen p50 s | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt | 68 | 0% | 1.87 | 0.597 [0.48, 0.71] | 0.594 [0.48, 0.71] | 0.835 [0.73, 0.92] | 0.945 [0.90, 0.98] | 0.573 [0.44, 0.71] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.990 [0.97, 1.00] | 0.328 [0.21, 0.44] | 0.571 [0.14, 0.86] |
| nano | 68 | 1% | 3.36 | 0.643 [0.53, 0.75] | 0.604 [0.48, 0.72] | 0.812 [0.70, 0.91] | 0.994 [0.98, 1.00] | 0.564 [0.43, 0.70] | 0.938 [0.84, 1.00] | 0.074 [0.00, 0.19] | 1.000 [1.00, 1.00] | 0.267 [0.17, 0.38] | 1.000 [1.00, 1.00] |
| ultra | 68 | 0% | 6.48 | 0.650 [0.54, 0.76] | 0.643 [0.52, 0.76] | 0.884 [0.80, 0.95] | 0.971 [0.94, 0.99] | 0.667 [0.52, 0.79] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.295 [0.18, 0.41] | 1.000 [1.00, 1.00] |

**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: ultra**