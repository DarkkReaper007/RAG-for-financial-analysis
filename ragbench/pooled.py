"""Confirmatory pooled analysis over dev + test items (decisions were already fixed on dev; this only adds power).
    python -m ragbench.pooled   ->  ragbench/results/report__pooled.md"""
import json

from ragbench import ablate as A
from ragbench.runner import RESULTS, flat
from ragbench.stats import holm, mcnemar_exact, mean_ci, paired_bootstrap


def pooled_rows(study, lvl, mode, gen=None):
    out = {}
    for split in ("dev", "test"):
        try:
            rows = A.load_rows(A.run_name(study, lvl, mode, gen, split), split, mode)
        except FileNotFoundError:
            continue
        out.update({f"{split}:{i}": r for i, r in rows.items()})
    return out


def table(study, mode, gen=None):
    prim = (A.GEN_PRIMARY if mode == "generation" else A.PRIMARY)[study]
    binm = (A.GEN_BINARY if mode == "generation" else A.BINARY)[study]
    levels = A.GEN_LEVELS.get(study, []) if mode == "generation" else [l for l in A.ABLATION_LEVELS[study] if l != A.BASE_LEVEL[study]]
    B = pooled_rows(study, A.BASE_LEVEL[study], mode, gen)
    if not B:
        return []
    ids = list(B)
    title = f"{study}" + (f" ({mode}, {gen})" if mode == "generation" else " (retrieval)")
    lines = [f"### {title}: pooled n={len(ids)} ({sum(i.startswith('dev') for i in ids)} dev + {sum(i.startswith('test') for i in ids)} test)", "",
             "| level | " + " | ".join(prim) + f" | Δ {prim[0]} vs base [95% CI] | McNemar p ({binm}, Holm) |", "|---|" + "---|" * (len(prim) + 2)]
    stats, rows = {}, [(A.BASE_LEVEL[study], B)]
    for l in levels:
        R = pooled_rows(study, l, mode, gen)
        if not R:
            continue
        rows.append((l, R))
        d = paired_bootstrap([flat(R.get(i, {})).get(prim[0]) for i in ids], [flat(B[i]).get(prim[0]) for i in ids])
        p, n10, n01 = mcnemar_exact([flat(R.get(i, {})).get(binm) for i in ids], [flat(B[i]).get(binm) for i in ids])
        stats[l] = (d, p, n10, n01)
    adj = dict(zip(stats, holm([s[1] for s in stats.values()]))) if stats else {}
    for l, R in rows:
        cells = [A.fmt(*mean_ci([flat(R[i]).get(m) for i in ids if i in R])) for m in prim]
        if l == A.BASE_LEVEL[study]:
            lines.append(f"| **{l}** (baseline) | " + " | ".join(cells) + " | - | - |")
        else:
            d, p, n10, n01 = stats[l]
            delta = f"{d['diff']:+.3f} [{d['lo']:.2f}, {d['hi']:.2f}]" if d["lo"] is not None else str(d["diff"])
            lines.append(f"| {l} | " + " | ".join(cells) + f" | {delta} | {adj[l]:.3f} (wins {n10} / losses {n01}) |")
    return lines + [""]


def main():
    out = ["# Pooled confirmatory analysis (dev + test)", "",
           "Decisions were fixed on dev with pre-registered rules; pooling adds statistical power and does not change them. Gold set: LLM-generated, filtered and LLM-verified, then treated as audited by the project author (no per-item human decision log is committed).", ""]
    for s in ("3_retrieval_mode", "1_chunking_hierarchy", "2_parsing_tables"):
        out += table(s, "retrieval")
    for g in ("nano", "bolt"):
        out += table("4_verification", "generation", g) if g == "nano" else []
        out += table("1_chunking_hierarchy", "generation", g)
    (RESULTS / "report__pooled.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
