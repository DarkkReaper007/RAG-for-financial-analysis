"""Run one ablation study (one factor varied against the baseline) and write a paired-statistics report.

  python -m ragbench.ablate --study 3_retrieval_mode --split dev
  python -m ragbench.ablate --study 1_chunking_hierarchy --split dev --plan      # show which indexes must be built
Each level is a runner invocation named <study>__<level>; indexes for new ingest configs are built (and cached) on first use.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ipo_rag.config import ABLATION_LEVELS, BASELINE, with_changes
from ipo_rag.indexing import index_dir
from ipo_rag.corpus import manifest
from ragbench.runner import RESULTS, flat, run_dir
from ragbench.stats import holm, mcnemar_exact, mean_ci, paired_bootstrap

BASE_LEVEL = {"1_chunking_hierarchy": "section_parent", "2_parsing_tables": "markdown", "3_retrieval_mode": "hybrid", "4_verification": "off", "5_embedder": "bge-small"}
PRIMARY = {"1_chunking_hierarchy": ["final.recall@5", "final.ndcg@5", "ctx.recall@10"],
           "2_parsing_tables": ["final.recall@5", "final.ndcg@5", "ctx.recall@10"],
           "3_retrieval_mode": ["cand.recall@5", "cand.ndcg@5", "cand.mrr"],
           "5_embedder": ["final.recall@5", "final.ndcg@5", "cand.recall@10"]}
BINARY = {"1_chunking_hierarchy": "final.hit@5", "2_parsing_tables": "final.hit@5", "3_retrieval_mode": "cand.hit@5", "5_embedder": "final.hit@5"}
SUBSET = {"2_parsing_tables": {"table", "numeric"}}       # where the factor should matter most


def missing_indexes(study):
    out = []
    for lvl, kw in ABLATION_LEVELS[study].items():
        cfg = with_changes(BASELINE, **kw)
        miss = [d for d in manifest() if not (index_dir(cfg.ingest, d) / "meta.json").exists()]
        if miss:
            out.append((lvl, cfg.ingest.key(), len(miss)))
    return out


def load_rows(name, split, mode, oracle):
    p = run_dir(name, split, mode, oracle) / "rows.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))}


def fmt(m, lo, hi):
    return "-" if m is None else f"{m:.3f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")


def report(study, split, mode, oracle):
    base = f"{study}__{BASE_LEVEL[study]}"
    B = load_rows(base, split, mode, oracle)
    levels = [l for l in ABLATION_LEVELS[study] if l != BASE_LEVEL[study]]
    lines = [f"## {study} ({split}, {mode}{', oracle' if oracle else ', open corpus'}); baseline = {BASE_LEVEL[study]}", ""]
    for subset_name, types in [("all items", None)] + ([(" + ".join(sorted(SUBSET[study])) + " only", SUBSET[study])] if study in SUBSET else []):
        ids = [i for i, r in B.items() if types is None or r["type"] in types]
        lines += [f"### {subset_name} (n={len(ids)})", "", "| level | " + " | ".join(PRIMARY[study]) + " | Δ " + PRIMARY[study][0] + " vs base [95% CI] | McNemar p (" + BINARY[study] + ", Holm) |",
                  "|---|" + "---|" * (len(PRIMARY[study]) + 2)]
        stats, rows_out = {}, []
        for lvl in [BASE_LEVEL[study]] + levels:
            R = B if lvl == BASE_LEVEL[study] else load_rows(f"{study}__{lvl}", split, mode, oracle)
            cells = [fmt(*mean_ci([flat(R[i]).get(m) for i in ids if i in R])) for m in PRIMARY[study]]
            if lvl != BASE_LEVEL[study]:
                a = [flat(R[i]).get(PRIMARY[study][0]) for i in ids]
                b = [flat(B[i]).get(PRIMARY[study][0]) for i in ids]
                d = paired_bootstrap(a, b)
                p, n10, n01 = mcnemar_exact([flat(R[i]).get(BINARY[study]) for i in ids], [flat(B[i]).get(BINARY[study]) for i in ids])
                stats[lvl] = (d, p, n10, n01)
            rows_out.append((lvl, cells))
        adj = dict(zip(stats, holm([s[1] for s in stats.values()]))) if stats else {}
        for lvl, cells in rows_out:
            if lvl == BASE_LEVEL[study]:
                lines.append(f"| **{lvl}** (baseline) | " + " | ".join(cells) + " | - | - |")
            else:
                d, p, n10, n01 = stats[lvl]
                lines.append(f"| {lvl} | " + " | ".join(cells) + f" | {d['diff']:+.3f} [{d['lo']:.2f}, {d['hi']:.2f}] | {adj[lvl]:.3f} (wins {n10} / losses {n01}) |" if d["lo"] is not None
                             else f"| {lvl} | " + " | ".join(cells) + f" | {d['diff']} | {adj[lvl]:.3f} |")
        lines.append("")
    text = "\n".join(lines)
    (RESULTS / f"report__{study}__{split}__{mode}{'__oracle' if oracle else ''}.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(ABLATION_LEVELS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="retrieval")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.plan:
        for lvl, key, n in missing_indexes(a.study) or [("(none)", "", 0)]:
            print(f"index to build: level={lvl} ingest_key={key} docs_missing={n}")
        return
    if not a.report_only:
        for lvl, kw in ABLATION_LEVELS[a.study].items():
            cmd = [sys.executable, "-m", "ragbench.runner", "--name", f"{a.study}__{lvl}", "--split", a.split, "--mode", a.mode, "--set"] + [f"{k}={v}" for k, v in kw.items()]
            subprocess.run(cmd + (["--oracle"] if a.oracle else []), check=True, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    print(report(a.study, a.split, a.mode, a.oracle))


if __name__ == "__main__":
    main()
