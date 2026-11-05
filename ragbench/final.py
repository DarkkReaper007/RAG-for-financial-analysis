"""Final stage: pre-registered decisions -> final configuration -> leave-one-out (dev) -> one-shot test run -> report.

Adoption rules (written before the runs): a non-baseline level replaces the baseline only if it improves the study's primary metric with the
paired 95% bootstrap CI above zero, or Holm-adjusted McNemar p < 0.05 in its favour; otherwise the simpler baseline is kept.
Verification is adopted only if it reduces answers with unsupported figures (Holm p < 0.05) without lowering judge.overall by more than 0.03.
Hierarchy (flat vs parent) is decided on answer-level judge.overall. The generator comes from generator_choice.json.
    python -m ragbench.final [--deadline <epoch seconds>]
"""
import argparse
import json
import time
from collections import Counter

from ipo_rag.config import ABLATION_LEVELS, BASELINE, with_changes
from ragbench import ablate as A
from ragbench.runner import RESULTS, flat, run_dir
from ragbench.stats import holm, mcnemar_exact, mean_ci, paired_bootstrap


def level_stats(study, split, mode, gen=None):
    """{level: {"diff", "lo", "hi", "p_adj", "n10", "n01"}} versus the study baseline on the study's first primary metric / binary metric."""
    B, levels, prim, binm = A.compare_levels(study, split, mode, False, gen)
    out, ps = {}, []
    for lvl in levels:
        R = A.load_rows(A.run_name(study, lvl, mode, gen), split, mode)
        ids = list(B)
        d = paired_bootstrap([flat(R.get(i, {})).get(prim[0]) for i in ids], [flat(B[i]).get(prim[0]) for i in ids])
        p, n10, n01 = mcnemar_exact([flat(R.get(i, {})).get(binm) for i in ids], [flat(B[i]).get(binm) for i in ids])
        out[lvl] = {**d, "p": p, "n10": n10, "n01": n01}
        ps.append(p)
    for lvl, pa in zip(out, holm(ps)):
        out[lvl]["p_adj"] = pa
    return out


def adopt(st):
    return st["diff"] is not None and st["diff"] > 0 and ((st["lo"] is not None and st["lo"] > 0) or (st["p_adj"] < 0.05 and st["n10"] > st["n01"]))


def decide(split="dev"):
    dec, why = {}, {}
    gen = json.load(open(RESULTS / "generator_choice.json")).get("chosen") if (RESULTS / "generator_choice.json").exists() else None
    gen = gen if gen in ("bolt", "nano") else "bolt"          # answer-level ablations were run with bolt and nano

    def pick(study, mapping, mode="retrieval", g=None):
        try:
            st = level_stats(study, split, mode, g)
        except FileNotFoundError:
            why[study] = "runs missing; baseline kept"
            return
        best = max((l for l in st if adopt(st[l])), key=lambda l: st[l]["diff"], default=None)
        why[study] = {l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for l, s in st.items()} | {"adopted": best or "baseline"}
        if best:
            dec.update(mapping[best])

    pick("3_retrieval_mode", {"dense": {"retrieval_mode": "dense"}, "bm25": {"retrieval_mode": "bm25"}})
    pick("2_parsing_tables", {"inline": {"table_mode": "inline"}, "flat": {"table_mode": "flat"}})
    pick("1_chunking_hierarchy", {"fixed_flat": {"chunker": "fixed", "hierarchy": "flat"}, "section_parent_ctx": {"contextual_header": True}})
    try:                                                           # hierarchy at answer level
        st = level_stats("1_chunking_hierarchy", split, "generation", gen)
        why["hierarchy(answer-level, " + gen + ")"] = {l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for l, s in st.items()}
        if adopt(st["section_flat"]):
            dec["hierarchy"] = "flat"
    except FileNotFoundError:
        why["hierarchy"] = "answer-level runs missing; parent kept"
    try:                                                           # verification: fewer unsupported figures, judge not worse
        st = level_stats("4_verification", split, "generation", gen)["on"]
        why["verification(" + gen + ")"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}
        B = A.load_rows(A.run_name("4_verification", "off", "generation", gen), split, "generation")
        R = A.load_rows(A.run_name("4_verification", "on", "generation", gen), split, "generation")
        d = paired_bootstrap([flat(R.get(i, {})).get("judge.overall") for i in B], [flat(B[i]).get("judge.overall") for i in B])
        why["verification(" + gen + ")"]["judge_overall_diff"] = d["diff"]
        if st["p_adj"] < 0.05 and st["n01"] > st["n10"] and (d["diff"] or 0) >= -0.03:
            dec["verify"] = "on"
    except FileNotFoundError:
        why["verification"] = "runs missing; off kept"
    try:                                                           # embedder (dev corpus): candidate only, needs a full-corpus index to be usable
        st = level_stats("5_embedder", split, "retrieval")
        cand = max((l for l in st if adopt(st[l])), key=lambda l: st[l]["diff"], default=None)
        why["5_embedder"] = {l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for l, s in st.items()} | {"candidate": cand or "bge-small"}
        if cand:
            dec["embedder"] = cand
    except FileNotFoundError:
        why["5_embedder"] = "runs missing; bge-small kept"
    return dec, why, gen


def index_ready(overrides):
    cfg = with_changes(BASELINE, **overrides)
    from ipo_rag.corpus import manifest
    from ipo_rag.indexing import index_dir
    return all((index_dir(cfg.ingest, d) / "meta.json").exists() for d in manifest())


def kv(d):
    return [f"{k}={v}" for k, v in d.items()]


def taxonomy(rows):
    c = Counter()
    for r in rows:
        if not r["answerable"]:
            c["unanswerable: correctly abstained" if (r.get("grounding") or {}).get("correct_abstention") else "unanswerable: answered anyway (hallucination)"] += 1
        elif r["failed"]:
            c["generation failed"] += 1
        elif (r.get("grounding") or {}).get("false_abstention"):
            c["abstained although evidence exists" if (flat(r).get("final.recall@5") or 0) > 0 else "abstained, evidence not retrieved"] += 1
        elif (flat(r).get("ctx.recall@10") or 0) == 0:
            c["answered, evidence not retrieved"] += 1
        elif (flat(r).get("judge.overall") or 0) >= 0.75:
            c["good answer"] += 1
        else:
            c["evidence retrieved but answer poor"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, default=0, help="epoch seconds after which long index builds are skipped")
    a = ap.parse_args()
    dec, why, gen = decide("dev")
    gm = A.GENS[gen] if (RESULTS / "generator_choice.json").exists() else A.GENS["bolt"]
    gen = json.load(open(RESULTS / "generator_choice.json")).get("chosen") or "bolt" if (RESULTS / "generator_choice.json").exists() else "bolt"
    gm = A.GENS[gen]
    final_ov = dict(dec)
    if "embedder" in final_ov and not index_ready({k: v for k, v in final_ov.items() if k in ("embedder", "table_mode", "chunker", "hierarchy", "contextual_header")}):
        need_h = 4.0
        if a.deadline and time.time() + need_h * 3600 > a.deadline:
            why["embedder_note"] = f"{final_ov['embedder']} was best on the dev corpus but a full 12-filing index (~{need_h:.0f} h) would not finish before the deadline; not used for the test run"
            final_ov.pop("embedder")
        else:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "ipo_rag.cli", "ingest", "--set"] + kv({k: v for k, v in final_ov.items() if k in ("embedder", "table_mode", "chunker", "hierarchy", "contextual_header")}))
            if not index_ready(final_ov):
                why["embedder_note"] = "full index build failed; embedder not used"
                final_ov.pop("embedder")
    json.dump({"overrides_vs_baseline": final_ov, "generator": gen, "generator_model": gm, "rationale": why}, open(RESULTS / "final_config.json", "w"), indent=1, default=str)
    print("FINAL CONFIG overrides:", final_ov or "(baseline)", "| generator:", gen, flush=True)

    # E8 leave-one-out (retrieval, dev, open corpus): replace each component by its simplest alternative
    ablate_to = {"retrieval_mode": "dense", "chunker": "fixed", "table_mode": "inline", "embedder": "bge-small", "contextual_header": False}
    base_ov = {"retrieval_mode": "hybrid", "chunker": "section", "table_mode": "markdown", "embedder": "bge-small", "contextual_header": False}
    full = {**base_ov, **final_ov}
    loo = {"final": {}}
    A.run_one("loo__final", "dev", "retrieval", kv(final_ov))
    for f, simple in ablate_to.items():
        if full.get(f) == simple:
            continue
        ov = {**final_ov, f: simple} | ({"hierarchy": "flat"} if f == "chunker" else {})
        if not index_ready(ov):
            loo[f] = "skipped (index for this combination not built)"
            continue
        if A.run_one(f"loo__{f}", "dev", "retrieval", kv(ov)):
            loo[f] = "done"
    lines = ["## Leave-one-out (dev, open corpus): final configuration vs final with one component replaced", "", "| component replaced | final.recall@5 | Δ vs final [95% CI] | McNemar p (final.hit@5) |", "|---|---|---|---|"]
    Fr = A.load_rows("loo__final", "dev", "retrieval")
    lines.append("| (none: final config) | " + A.fmt(*mean_ci([flat(Fr[i]).get("final.recall@5") for i in Fr])) + " | - | - |")
    for f, s in loo.items():
        if f == "final" or s != "done":
            if f != "final":
                lines.append(f"| {f} -> {ablate_to[f]} | {s} | | |")
            continue
        R = A.load_rows(f"loo__{f}", "dev", "retrieval")
        d = paired_bootstrap([flat(R[i]).get("final.recall@5") for i in Fr], [flat(Fr[i]).get("final.recall@5") for i in Fr])
        p, n10, n01 = mcnemar_exact([flat(R[i]).get("final.hit@5") for i in Fr], [flat(Fr[i]).get("final.hit@5") for i in Fr])
        lines.append(f"| {f} -> {ablate_to[f]} | {A.fmt(*mean_ci([flat(R[i]).get('final.recall@5') for i in Fr]))} | {d['diff']:+.3f} [{d['lo']:.2f}, {d['hi']:.2f}] | {p:.3f} |" if d["lo"] is not None else f"| {f} -> {ablate_to[f]} | | | |")
    (RESULTS / "report__leave_one_out.md").write_text("\n".join(lines))

    # E9 / E10: one-shot test runs (gold set treated as author-audited)
    ov_final = kv(final_ov) + [f"model={gm}"] + A.GEN_COMMON
    ov_base = [f"model={gm}"] + A.GEN_COMMON
    runs = {"final": ov_final}
    if final_ov:
        runs["baseline"] = ov_base
    for tag, ov in runs.items():
        name = f"test_{tag}__{gen}"
        if A.run_one(name, "test", "generation", ov):
            A.judge_run(run_dir(name, "test", "generation", False).name)
    A.run_one("test_final_oracle", "test", "retrieval", kv(final_ov), oracle=True)

    # report
    keys = ["final.recall@5", "final.ndcg@5", "final.full_hit@5", "ctx.recall@10", "judge.overall", "judge.coverage", "judge.faithfulness", "judge.redundancy", "grounding.key_fact_recall",
            "grounding.numeric_support", "grounding.unsupported_figure", "grounding.citation_valid", "grounding.false_abstention", "grounding.correct_abstention"]
    out = ["# Final test-split results (gold set: LLM-generated, filtered and LLM-verified, then treated as audited by the project author (no per-item human decision log is committed))", "", f"Generator: **{gen}** (`{gm}`); judge: gpt-oss-120b (gold-vs-prediction). Final-config overrides vs baseline: `{final_ov or 'none (baseline)'}`", ""]
    for tag in runs:
        R = [json.loads(l) for l in open(run_dir(f"test_{tag}__{gen}", "test", "generation", False) / "rows.judged.jsonl")]
        out += [f"## {tag} (n={len(R)}, failure rate {sum(r['failed'] for r in R) / len(R):.0%})", "", "| group | n | " + " | ".join(k.split('.')[1] + ("" if not k.startswith("judge") else "") for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
        for g in ["ALL"] + sorted({r["type"] for r in R}):
            S = R if g == "ALL" else [r for r in R if r["type"] == g]
            out.append(f"| {g} | {len(S)} | " + " | ".join(A.fmt(*mean_ci([flat(r).get(k) for r in S])) for k in keys) + " |")
        out += ["", "Failure taxonomy: " + "; ".join(f"{k}: {v}" for k, v in taxonomy(R).most_common()), ""]
        worst = sorted([r for r in R if r["answerable"] and not r["failed"]], key=lambda r: flat(r).get("judge.overall") if flat(r).get("judge.overall") is not None else 0)[:8]
        out += ["Lowest-scoring answerable items: " + ", ".join(f"{r['id']} ({r['type']}, judge {flat(r).get('judge.overall')})" for r in worst), ""]
    if len(runs) == 2:
        Fr_, Br_ = (A.load_rows(f"test_{t}__{gen}", "test", "generation") for t in ("final", "baseline"))
        d = paired_bootstrap([flat(Fr_.get(i, {})).get("judge.overall") for i in Br_], [flat(Br_[i]).get("judge.overall") for i in Br_])
        out += [f"Paired final vs baseline judge.overall: {d['diff']:+.3f} [{d['lo']}, {d['hi']}] (n={d['n']})", ""]
    O = A.load_rows("test_final_oracle", "test", "retrieval", True)
    out += ["## Oracle retrieval (each question restricted to its own filing(s)), final configuration", "", "| metric | value |", "|---|---|"]
    out += [f"| {k} | {A.fmt(*mean_ci([flat(r).get(k) for r in O.values()]))} |" for k in ("final.recall@5", "final.ndcg@5", "final.full_hit@5", "cand.recall@10")]
    (RESULTS / "final_report.md").write_text("\n".join(out))
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
