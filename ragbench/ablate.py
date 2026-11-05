"""Run ablation studies and write paired-statistics reports.

Retrieval studies (no LLM):   python -m ragbench.ablate --study 3_retrieval_mode --split dev
Answer-level studies:         python -m ragbench.ablate --study 4_verification --mode generation --gen bolt
Generator comparison:         python -m ragbench.ablate --generators bolt nano ultra --split dev
Runs are idempotent (finished runs are skipped). Generation runs are judged with ragbench.judge (gold-vs-prediction).
"""
import argparse
import json
import os
import subprocess
import sys

from ipo_rag.config import ABLATION_LEVELS, BASELINE, with_changes
from ipo_rag.corpus import manifest
from ipo_rag.indexing import index_dir
from ragbench.runner import RESULTS, flat, run_dir
from ragbench.stats import holm, mcnemar_exact, mean_ci, paired_bootstrap

GENS = {"bolt": "nvidia/nemotron-3.5-bolt-e4b", "nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "ultra": "nvidia/nemotron-3-ultra-550b-a55b"}
BASE_LEVEL = {"1_chunking_hierarchy": "section_parent", "2_parsing_tables": "markdown", "3_retrieval_mode": "hybrid", "4_verification": "off", "5_embedder": "bge-small"}
CORPUS = {"5_embedder": "dev"}                                  # study 5 indexes only the dev filings (compute)
GEN_LEVELS = {"1_chunking_hierarchy": ["section_flat"], "4_verification": ["on"]}   # answer-level comparisons
PRIMARY = {"1_chunking_hierarchy": ["final.recall@5", "final.ndcg@5", "ctx.recall@10"], "2_parsing_tables": ["final.recall@5", "final.ndcg@5", "ctx.recall@10"],
           "3_retrieval_mode": ["cand.recall@5", "cand.ndcg@5", "cand.mrr"], "5_embedder": ["final.recall@5", "final.ndcg@5", "cand.recall@10"]}
BINARY = {"1_chunking_hierarchy": "final.hit@5", "2_parsing_tables": "final.hit@5", "3_retrieval_mode": "cand.hit@5", "5_embedder": "final.hit@5"}
GEN_PRIMARY = {"1_chunking_hierarchy": ["judge.overall", "judge.coverage", "judge.faithfulness", "grounding.key_fact_recall", "grounding.numeric_support"],
               "4_verification": ["judge.overall", "grounding.unsupported_figure", "grounding.false_abstention", "judge.faithfulness", "grounding.citation_valid"]}
GEN_BINARY = {"1_chunking_hierarchy": "grounding.key_fact_recall", "4_verification": "grounding.unsupported_figure"}
LOWER_IS_BETTER = {"grounding.unsupported_figure", "grounding.false_abstention"}
SUBSET = {"2_parsing_tables": {"table", "numeric"}}
GEN_COMMON = ["max_tokens=4096"]          # reasoning models (nano) exhausted 2048 tokens before answering -> empty replies


def run_name(study, lvl, mode, gen=None, split="dev"):
    if mode == "generation":
        return f"genbase__{gen}" if lvl == BASE_LEVEL[study] else f"{study}__{lvl}__{gen}"
    return f"{study}__{lvl}" + (f"__{split}corpus" if study in CORPUS else "")


def missing_indexes(study):
    docs = [d for d, m in manifest().items() if study not in CORPUS or m["split"] == "dev"]
    out = []
    for lvl, kw in ABLATION_LEVELS[study].items():
        cfg = with_changes(BASELINE, **kw)
        miss = [d for d in docs if not (index_dir(cfg.ingest, d) / "meta.json").exists()]
        if miss:
            out.append((lvl, cfg.ingest.key(), len(miss)))
    return out


def sh(cmd, log=True):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    return subprocess.run(cmd, env=env, check=False).returncode


def run_one(name, split, mode, overrides, corpus="all", oracle=False, force=False):
    d = run_dir(name, split, mode, oracle)
    if (d / "metrics.json").exists() and not force:
        print("exists:", d.name, flush=True)
        return True
    cmd = [sys.executable, "-m", "ragbench.runner", "--name", name, "--split", split, "--mode", mode, "--corpus", corpus, "--set"] + overrides
    if oracle:
        cmd.append("--oracle")
    rc = sh(cmd)
    return rc == 0 and (d / "metrics.json").exists()


def judge_run(dirname):
    d = RESULTS / dirname
    if (d / "metrics.judged.json").exists():
        return True
    sh([sys.executable, "-m", "ragbench.judge", "--run", dirname])
    return (d / "metrics.judged.json").exists()


def load_rows(name, split, mode, oracle=False):
    d = run_dir(name, split, mode, oracle)
    p = d / ("rows.judged.jsonl" if mode == "generation" else "rows.jsonl")
    return {r["id"]: r for r in map(json.loads, open(p))}


def fmt(m, lo, hi):
    return "-" if m is None else f"{m:.3f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")


def compare_levels(study, split, mode, oracle=False, gen=None):
    """-> {level: {"cells": [...], "delta": {...}|None, "p_adj": float|None, "n10", "n01"}} for baseline + each level, for all items."""
    levels = GEN_LEVELS.get(study, []) if mode == "generation" else [l for l in ABLATION_LEVELS[study] if l != BASE_LEVEL[study]]
    prim = (GEN_PRIMARY if mode == "generation" else PRIMARY)[study]
    binm = (GEN_BINARY if mode == "generation" else BINARY)[study]
    B = load_rows(run_name(study, BASE_LEVEL[study], mode, gen, split), split, mode, oracle)
    return B, levels, prim, binm


def report(study, split, mode="retrieval", oracle=False, gen=None):
    B, levels, prim, binm = compare_levels(study, split, mode, oracle, gen)
    tag = f"{mode}{'__oracle' if oracle else ''}{'__' + gen if gen else ''}"
    lines = [f"## {study} ({split}, {mode}{', ' + gen if gen else ''}{', oracle' if oracle else ', open corpus'}); baseline = {BASE_LEVEL[study]}", ""]
    if mode == "generation" and study in LOWER_NOTE:
        lines += [LOWER_NOTE[study], ""]
    for subset_name, types in [("all items", None)] + ([(" + ".join(sorted(SUBSET[study])) + " only", SUBSET[study])] if study in SUBSET and mode == "retrieval" else []):
        ids = [i for i, r in B.items() if types is None or r["type"] in types]
        lines += [f"### {subset_name} (n={len(ids)})", "", "| level | " + " | ".join(prim) + f" | Δ {prim[0]} vs base [95% CI] | McNemar p ({binm}, Holm) |", "|---|" + "---|" * (len(prim) + 2)]
        stats, rows_out = {}, []
        for lvl in [BASE_LEVEL[study]] + levels:
            R = B if lvl == BASE_LEVEL[study] else load_rows(run_name(study, lvl, mode, gen, split), split, mode, oracle)
            cells = [fmt(*mean_ci([flat(R[i]).get(m) for i in ids if i in R])) for m in prim]
            if lvl != BASE_LEVEL[study]:
                a = [flat(R.get(i, {})).get(prim[0]) for i in ids]
                b = [flat(B[i]).get(prim[0]) for i in ids]
                d = paired_bootstrap(a, b)
                p, n10, n01 = mcnemar_exact([flat(R.get(i, {})).get(binm) for i in ids], [flat(B[i]).get(binm) for i in ids])
                stats[lvl] = (d, p, n10, n01)
            rows_out.append((lvl, cells))
        adj = dict(zip(stats, holm([s[1] for s in stats.values()]))) if stats else {}
        for lvl, cells in rows_out:
            if lvl == BASE_LEVEL[study]:
                lines.append(f"| **{lvl}** (baseline) | " + " | ".join(cells) + " | - | - |")
            else:
                d, p, n10, n01 = stats[lvl]
                delta = f"{d['diff']:+.3f} [{d['lo']:.2f}, {d['hi']:.2f}]" if d["lo"] is not None else str(d["diff"])
                lines.append(f"| {lvl} | " + " | ".join(cells) + f" | {delta} | {adj[lvl]:.3f} (wins {n10} / losses {n01}) |")
        lines.append("")
    text = "\n".join(lines)
    (RESULTS / f"report__{study}__{split}__{tag}.md").write_text(text)
    return text


LOWER_NOTE = {"4_verification": "*Lower is better for `unsupported_figure` (answers containing a figure absent from the retrieved text) and `false_abstention`.*"}


def run_study(study, split, mode, gen=None, oracle=False, judge=True):
    levels = ABLATION_LEVELS[study]
    if mode == "generation":
        names = [BASE_LEVEL[study]] + GEN_LEVELS[study]
        for lvl in names:
            ov = [f"{k}={v}" for k, v in levels[lvl].items()] + [f"model={GENS[gen]}"] + GEN_COMMON
            name = run_name(study, lvl, mode, gen, split)
            if run_one(name, split, mode, ov, split if study in CORPUS else "all") and judge:
                judge_run(run_dir(name, split, mode, False).name)
    else:
        for lvl, kw in levels.items():
            run_one(run_name(study, lvl, mode, None, split), split, mode, [f"{k}={v}" for k, v in kw.items()], split if study in CORPUS else "all", oracle)


def generator_comparison(gens, split):
    rows = {}
    for g in gens:
        name = f"genbase__{g}"
        ov = [f"{k}={v}" for k, v in {}.items()] + [f"model={GENS[g]}"] + GEN_COMMON
        if run_one(name, split, "generation", ov):
            judge_run(run_dir(name, split, "generation", False).name)
        p = run_dir(name, split, "generation", False)
        if (p / "rows.judged.jsonl").exists():
            rows[g] = [json.loads(l) for l in open(p / "rows.judged.jsonl")]
    keys = ["judge.overall", "judge.coverage", "judge.faithfulness", "judge.redundancy", "grounding.key_fact_recall", "grounding.numeric_support",
            "grounding.unsupported_figure", "grounding.citation_valid", "grounding.false_abstention", "grounding.correct_abstention"]
    lines = [f"## Generator comparison ({split}, baseline pipeline config, judge gpt-oss-120b)", "",
             "| generator | n | failure rate | gen p50 s | " + " | ".join(k.split(".")[1] for k in keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    summary = {}
    for g, R in rows.items():
        fr = sum(r["failed"] for r in R) / len(R)
        lat = sorted(r["generation_s"] for r in R if not r["failed"])
        cells = [fmt(*mean_ci([flat(r).get(k) for r in R])) for k in keys]
        lines.append(f"| {g} | {len(R)} | {fr:.0%} | {lat[len(lat) // 2] if lat else '-'} | " + " | ".join(cells) + " |")
        summary[g] = {"failure_rate": fr, "overall": mean_ci([flat(r).get("judge.overall") for r in R])[0]}
    # pre-registered rule: highest mean judge.overall among generators with failure_rate <= 10%; ties -> cheaper (bolt < nano < ultra)
    ok = {g: s for g, s in summary.items() if s["overall"] is not None and s["failure_rate"] <= 0.10}
    choice = max(ok, key=lambda g: (round(ok[g]["overall"], 3), -list(GENS).index(g))) if ok else None
    lines += ["", f"**Pre-registered selection rule:** highest mean judge.overall among generators with failure rate <= 10% (ties -> cheaper). **Chosen: {choice}**"]
    text = "\n".join(lines)
    (RESULTS / f"report__generators__{split}.md").write_text(text)
    json.dump({"chosen": choice, "model": GENS.get(choice), "summary": summary, "rule": "max judge.overall among failure_rate<=0.10; ties -> cheaper"},
              open(RESULTS / ("generator_choice.json" if os.environ.get("RAG_CORPUS", "ipo") == "ipo" else f"generator_choice__{os.environ['RAG_CORPUS']}.json"), "w"), indent=1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(ABLATION_LEVELS))
    ap.add_argument("--generators", nargs="*")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="retrieval", choices=["retrieval", "generation"])
    ap.add_argument("--gen", default="bolt", choices=list(GENS))
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="generate only; judge later (the Groq free tier allows ~200k tokens/day/key for gpt-oss-120b)")
    ap.add_argument("--judge-pending", action="store_true", help="judge every generation run that has no judged output yet")
    a = ap.parse_args()
    if a.judge_pending:
        for d in sorted(RESULTS.glob("*__generation")):
            if (d / "metrics.json").exists() and not (d / "metrics.judged.json").exists():
                print("judging", d.name, flush=True)
                judge_run(d.name)
        return
    if a.generators:
        print(generator_comparison(a.generators, a.split))
        return
    if a.plan:
        for lvl, key, n in missing_indexes(a.study) or [("(none)", "", 0)]:
            print(f"index to build: level={lvl} ingest_key={key} docs_missing={n}")
        return
    if not a.report_only:
        run_study(a.study, a.split, a.mode, a.gen, a.oracle, judge=not a.no_judge)
    if a.no_judge and a.mode == "generation":
        return
    print(report(a.study, a.split, a.mode, a.oracle, a.gen if a.mode == "generation" else None))


if __name__ == "__main__":
    main()
