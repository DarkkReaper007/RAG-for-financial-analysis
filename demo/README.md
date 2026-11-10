# Prospectus Desk (demo)

A small web app over the real pipeline: ask a question about the 12 IPO filings, read the answer with page citations, see the source page with the figures highlighted, see how the system found the evidence, compare two settings side by side, and browse the experiment results.

## Run it

```bash
source .venv/bin/activate
export NVIDIA_API_KEYS=key1,key2   # needed to write NEW answers; set it in the same shell, never in a file
python -m demo.server              # http://127.0.0.1:8000  (needs the built indexes in indexes/ and the PDFs in data/prospectuses/)
```

The pill at the top right shows the server state:

| Pill | Meaning |
|---|---|
| Loading search models | Start-up. Takes about 25 s once; the first question waits for it |
| Answers live | A model key is set, so any question gets a written answer |
| Saved answers only | No key. Search still works and the example questions replay from `demo/cache/`; other questions show the best passages and a short explanation |

If a question fails, the screen says why (no key, rate limit, timeout, rejected key) and still shows the retrieved passages.

## Screens

**Ask** has the answer with citations, the source page viewer, and *How the system found it*: the five steps (search, combine, rerank, read, answer) with timings, and the passages that were used, each marked with which search found it and how the reranker moved it.
**Compare** runs two settings on the same question (hybrid vs meaning-only search, figure check on or off, with or without surrounding text).
**Results** shows the experiment charts from `results/`.

The six example questions come from the held-out test split and include failures on purpose (a wrong figure, a missed two-filing question, a false abstention).

## Check it before a demo

With the server running, in a second terminal:

```bash
python -m demo.smoke_test            # 12 questions x 5 settings, with timings
python -m demo.smoke_test --full     # every mode x figure check x surrounding text (144 calls; long with a key)
python -m demo.smoke_test --fresh    # ignore saved answers, so answer-writing time is real
```

It prints one line per call and a summary, and exits with 1 if anything failed unexpectedly. Typical times on a laptop: search 0.2 to 1 s (the first question over all 12 filings is slower until start-up has finished), writing an answer 5 to 15 s.

Other commands: `python -m demo.server --warm` pre-computes the six examples in four settings into `demo/cache/`. Keys are read from the environment only and are never logged or stored.
