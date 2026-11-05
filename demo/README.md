# Prospectus Desk (demo)

A small web app over the real pipeline: ask a question about the 12 IPO filings, read the answer with page citations, see the source page with the figures highlighted, inspect how retrieval found it, compare two settings side by side, and browse the experiment results.

```bash
source .venv/bin/activate
python -m demo.server              # http://127.0.0.1:8000  (needs the built indexes in indexes/ and the PDFs in data/prospectuses/)
export NVIDIA_API_KEYS=key1,key2   # only for questions that are not already in demo/cache
python -m demo.server --warm       # pre-compute the six example questions in four settings
```

Screens: **Ask** (answer, citations, source page viewer, retrieval trace), **Compare** (hybrid vs meaning-only search, figure check on/off, with/without surrounding text), **Results** (charts from `results/`).
Saved answers in `demo/cache/` replay when the API is unreachable, so the demo works offline for the example questions. Keys are read from the environment only.
The six examples come from the held-out test split and include failures on purpose (a wrong figure, a missed two-filing question, a false abstention).
