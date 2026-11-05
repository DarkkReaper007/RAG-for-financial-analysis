## Ceiling (generator given the GOLD evidence pages) vs end-to-end (baseline retrieval), dev, answerable items

judge.overall: failed generations are excluded from each mean; the paired difference uses items answered in both runs.

| generator | group | n | end-to-end | ceiling (perfect retrieval) | retrieval loss (ceiling - end-to-end) [95% CI] |
|---|---|---|---|---|---|
| nano | ALL | 48 | 0.59 | 0.84 | +0.27 [+0.10, +0.43] |
| nano | definition | 3 | 0.64 | 0.92 | +0.28 |
| nano | factoid | 10 | 0.53 | 0.83 | +0.30 [-0.17, +0.70] |
| nano | list | 10 | 0.75 | 0.72 | +0.06 [-0.27, +0.38] |
| nano | multi_document | 5 | 0.20 | 0.60 | +0.40 [+0.00, +0.80] |
| nano | numeric | 10 | 0.56 | 0.93 | +0.37 [+0.11, +0.67] |
| nano | table | 10 | 0.73 | 1.00 | +0.25 [+0.00, +0.62] |
| bolt | ALL | 48 | 0.58 | 0.79 | +0.21 [+0.04, +0.38] |
| bolt | definition | 3 | 0.33 | 0.97 | +0.64 |
| bolt | factoid | 10 | 0.42 | 0.82 | +0.40 [-0.07, +0.80] |
| bolt | list | 10 | 0.71 | 0.72 | +0.01 [-0.28, +0.29] |
| bolt | multi_document | 5 | 0.00 | 0.75 | +0.75 [+0.35, +1.00] |
| bolt | numeric | 10 | 0.76 | 0.73 | -0.02 [-0.39, +0.33] |
| bolt | table | 10 | 0.79 | 0.87 | +0.08 [-0.17, +0.37] |
