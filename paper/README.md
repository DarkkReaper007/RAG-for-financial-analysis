# Report and slides

* `main.tex`: the report (LaTeX, `article`, standard packages only). Build with `pdflatex main.tex` twice (or upload `paper/` to Overleaf). It was not compiled in the environment where it was written, so check the page count (target 10 to 12) and any overfull boxes on first build.
* `slides.pptx`: the project talk, 17 slides (16 plus a backup) with speaker notes, built by `python paper/make_deck.py` (add `--preview` for PNG previews). Both decks share `deckkit.py`, which also runs a text-fit check.
* `figs/`: figures drawn by `python paper/make_figs.py` from `demo/results.json`; `demo_page.png` is a source page rendered by the demo server.
Every number in the report comes from `results/` (see `results/SUMMARY.md` and `results/report__financebench_analysis.md`).
* `pipeline_slides.pptx`: 14 slides that explain the pipeline stage by stage (no results), with speaker notes; built by `python paper/make_pipeline_deck.py` (add `--preview` for PNG previews in `paper/_preview/`).
