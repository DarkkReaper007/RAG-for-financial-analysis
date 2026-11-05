"""Assemble every report into ragbench/results/SUMMARY.md (re-run any time; reads whatever has finished)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
SECTIONS = [
    ("E1 Retrieval mode (study 3), dev", "report__3_retrieval_mode__dev__retrieval.md"),
    ("E1' Retrieval mode, TEST confirmation", "report__3_retrieval_mode__test__retrieval.md"),
    ("E2 Chunking & hierarchy, retrieval (study 1), dev", "report__1_chunking_hierarchy__dev__retrieval.md"),
    ("E2' Chunking & hierarchy, retrieval, TEST confirmation", "report__1_chunking_hierarchy__test__retrieval.md"),
    ("E3 Parsing & table representation (study 2), dev", "report__2_parsing_tables__dev__retrieval.md"),
    ("E3' Parsing & table representation, TEST confirmation", "report__2_parsing_tables__test__retrieval.md"),
    ("E4 Embedder (study 5), dev filings corpus", "report__5_embedder__dev__retrieval.md"),
    ("E4' Embedder, TEST filings corpus (confirmation)", "report__5_embedder__test__retrieval.md"),
    ("E5 Verification, answer level, nano, dev", "report__4_verification__dev__generation__nano.md"),
    ("E5 Verification, answer level, bolt, dev", "report__4_verification__dev__generation__bolt.md"),
    ("E5' Verification, nano, TEST confirmation", "report__4_verification__test__generation__nano.md"),
    ("E6 Hierarchy (parent vs flat), answer level, nano, dev", "report__1_chunking_hierarchy__dev__generation__nano.md"),
    ("E6 Hierarchy, answer level, bolt, dev", "report__1_chunking_hierarchy__dev__generation__bolt.md"),
    ("E6' Hierarchy, nano, TEST confirmation", "report__1_chunking_hierarchy__test__generation__nano.md"),
    ("E6' Hierarchy, bolt, TEST confirmation", "report__1_chunking_hierarchy__test__generation__bolt.md"),
    ("E7 Generator comparison, dev (selection rule applied here)", "report__generators__dev.md"),
    ("E7' Generator comparison, TEST confirmation", "report__generators__test.md"),
    ("Ceiling vs end-to-end (dev)", "report_ceiling_vs_endtoend.md"),
    ("E8 Leave-one-out, dev", "report__leave_one_out.md"),
    ("E9/E10 Final test-split run", "final_report.md"),
    ("Judge vs deterministic key-fact check (dev)", "report__judge_vs_deterministic.md"),
]
out = ["# Experiment summary (auto-generated)", "",
       "All results use gold_v1 (LLM-generated, filtered and LLM-verified, then treated as audited by the project author (no per-item human decision log is committed)). Dev = model/level selection; test = one-shot final run + confirmations.", ""]
fc = R / "final_config.json"
if fc.exists():
    j = json.load(open(fc))
    out += ["## Final configuration (pre-registered rules applied on dev)", "", f"* Overrides vs baseline: `{j['overrides_vs_baseline'] or 'none: baseline is the final configuration'}`",
            f"* Generator: `{j['generator_model']}`", ""]
for title, fn in SECTIONS:
    p = R / fn if (R / fn).exists() else R / fn.replace("report_ceiling", "report__ceiling")
    if p.exists():
        body = p.read_text().strip()
        body = "\n".join(l for l in body.splitlines() if not l.startswith("# "))
        out += [f"## {title}", "", body.replace("## ", "### ", 1) if body.startswith("## ") else body, ""]
(R / "SUMMARY.md").write_text("\n".join(out))
print("wrote", R / "SUMMARY.md", f"({len(out)} blocks)")
