# Final test-split results (gold set treated as audited by the authors)

Generator: **nano** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`); judge: gpt-oss-120b (gold-vs-prediction). Final-config overrides vs baseline: `none (baseline)`

## final (n=68, failure rate 1%)

| group | n | recall@5 | ndcg@5 | full_hit@5 | recall@10 | overall | coverage | faithfulness | redundancy | key_fact_recall | numeric_support | unsupported_figure | citation_valid | false_abstention | correct_abstention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 68 | 0.656 [0.54, 0.77] | 0.534 [0.43, 0.64] | 0.639 [0.52, 0.75] | 0.656 [0.54, 0.77] | 0.651 [0.54, 0.76] | 0.621 [0.50, 0.74] | 0.824 [0.72, 0.91] | 0.994 [0.98, 1.00] | 0.580 [0.44, 0.71] | 0.938 [0.84, 1.00] | 0.074 [0.00, 0.19] | 1.000 [1.00, 1.00] | 0.267 [0.17, 0.38] | 1.000 [1.00, 1.00] |
| definition | 3 | 0.333 | 0.333 | 0.333 | 0.333 | 0.639 | 0.667 | 0.875 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.333 | - |
| factoid | 13 | 0.769 [0.54, 1.00] | 0.630 [0.41, 0.84] | 0.769 [0.54, 1.00] | 0.769 [0.54, 1.00] | 0.590 [0.33, 0.82] | 0.538 [0.23, 0.77] | 0.778 [0.44, 1.00] | 1.000 [1.00, 1.00] | 0.455 [0.18, 0.73] | 0.750 | 0.250 | 1.000 [1.00, 1.00] | 0.308 [0.08, 0.54] | - |
| list | 11 | 0.727 [0.45, 1.00] | 0.553 [0.31, 0.78] | 0.727 [0.45, 1.00] | 0.727 [0.45, 1.00] | 0.758 [0.51, 0.94] | 0.750 [0.50, 0.95] | 0.889 [0.78, 0.97] | 0.972 [0.92, 1.00] | 0.667 | 0.667 | 0.500 | 1.000 [1.00, 1.00] | 0.182 [0.00, 0.45] | - |
| multi_document | 7 | 0.143 [0.00, 0.36] | 0.143 [0.00, 0.34] | 0.000 [0.00, 0.00] | 0.143 [0.00, 0.36] | 0.286 [0.00, 0.71] | 0.286 [0.00, 0.71] | 1.000 | 1.000 | 0.250 [0.00, 0.58] | - | - | 1.000 | 0.714 [0.29, 1.00] | - |
| numeric | 13 | 0.462 [0.23, 0.69] | 0.405 [0.17, 0.66] | 0.462 [0.23, 0.69] | 0.462 [0.23, 0.69] | 0.556 [0.31, 0.81] | 0.500 [0.25, 0.75] | 0.750 [0.38, 1.00] | 1.000 [1.00, 1.00] | 0.500 [0.25, 0.75] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.333 [0.08, 0.58] | - |
| table | 14 | 1.000 [1.00, 1.00] | 0.790 [0.66, 0.91] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.893 [0.76, 0.99] | 0.857 [0.64, 1.00] | 0.821 [0.62, 0.98] | 1.000 [1.00, 1.00] | 0.839 [0.64, 1.00] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | - |
| unanswerable | 7 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1.000 [1.00, 1.00] |

Failure taxonomy: good answer: 34; abstained, evidence not retrieved: 14; unanswerable: correctly abstained: 7; answered, evidence not retrieved: 6; evidence retrieved but answer poor: 4; abstained although evidence exists: 2; generation failed: 1

Lowest-scoring answerable items: t-fac-02 (factoid, judge 0.0), t-num-02 (numeric, judge 0.0), t-lis-01 (list, judge 0.0), t-lis-02 (list, judge 0.0), t-def-01 (definition, judge 0.0), t-fac-04 (factoid, judge 0.0), t-num-05 (numeric, judge 0.0), t-fac-06 (factoid, judge 0.0)

## Oracle retrieval (each question restricted to its own filing(s)), final configuration

| metric | value |
|---|---|
| final.recall@5 | 0.787 [0.68, 0.89] |
| final.ndcg@5 | 0.655 [0.56, 0.75] |
| final.full_hit@5 | 0.770 [0.66, 0.87] |
| cand.recall@10 | 0.803 [0.70, 0.89] |