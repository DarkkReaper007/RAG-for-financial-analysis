"""Figures for the report and the slides, drawn from demo/results.json (numbers copied from results/*.md). python paper/make_figs.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "figs"
D = json.load(open(Path(__file__).parent.parent / "demo" / "results.json"))
INK, MUTED, TEAL, GREY, LINE, BAD, WARN = "#14202b", "#5a6876", "#0b6e5f", "#9fb4c4", "#dbe1e7", "#b3261e", "#9a5b00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def retrieval_mode():
    rows = D["retrieval_mode"]["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    w = 0.26
    for j, (k, c, lab) in enumerate([("dev", GREY, "Dev (n=48)"), ("test", "#6f8ea5", "Test (n=61)"), ("pooled", TEAL, "Pooled (n=109)")]):
        xs = [i + (j - 1) * w for i in range(len(rows))]
        b = ax.bar(xs, [r[k] for r in rows], w, color=c, label=lab)
        for x, r in zip(xs, rows):
            ax.text(x, r[k] + 0.012, f"{r[k]:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_ylim(0, 0.75); ax.set_ylabel("recall@5 (first-stage candidates)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "retrieval_mode")


def forest():
    rows = D["ablations"]["rows"][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for i, r in enumerate(rows):
        c = TEAL if r.get("sig") else MUTED
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2)
        ax.plot(r["delta"], i, "o", color=c, ms=5.5)
    ax.axvline(0, color=LINE, lw=1.6, zorder=0)
    ax.set_yticks(range(len(rows)), [f"{r['study']}: {r['level']}" for r in rows])
    ax.set_xlabel("change in recall@5 versus baseline (95% interval)")
    ax.set_xlim(-0.22, 0.14)
    save(fig, "forest")


def ceiling():
    rows = D["ceiling"]["rows"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    w = 0.34
    for j, (k, c, lab) in enumerate([("nano", TEAL, "nano (chosen)"), ("bolt", GREY, "bolt")]):
        xs = [i + (j - 0.5) * w for i in range(2)]
        ax.bar(xs, [r[k] for r in rows], w, color=c, label=lab)
        for x, r in zip(xs, rows):
            ax.text(x, r[k] + 0.012, f"{r[k]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(2), ["Retrieved pages", "Gold pages given"]); ax.set_ylim(0, 1); ax.set_ylabel("judge score (dev)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "ceiling")


def by_type():
    rows = D["by_type"]["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    w = 0.38
    for j, (k, c, lab) in enumerate([("recall", TEAL, "recall@5"), ("judge", GREY, "judge score")]):
        xs = [i + (j - 0.5) * w for i in range(len(rows))]
        ax.bar(xs, [r[k] for r in rows], w, color=c, label=lab)
    ax.set_xticks(range(len(rows)), [f"{r['type']}\n(n={r['n']})" for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    save(fig, "by_type")


def taxonomy():
    rows = D["taxonomy"]["rows"][::-1]
    col = {"good": TEAL, "bad": BAD, "warn": WARN}
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.barh([r["label"] for r in rows], [r["n"] for r in rows], color=[col[r["tone"]] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r["n"] + 0.4, i, str(r["n"]), va="center", fontsize=8.5)
    ax.set_xlabel("test questions (of 68)")
    save(fig, "taxonomy")


def financebench():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    items = [("Hybrid", 0.333), ("Dense", 0.314), ("BM25", 0.179), ("Hybrid, right\nfiling only", 0.420)]
    ax.bar([i[0] for i in items], [i[1] for i in items], color=[TEAL, GREY, GREY, "#6f8ea5"], width=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.6); ax.set_ylabel("recall@5 (final)")
    save(fig, "financebench")


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.3)); ax.axis("off"); ax.set_xlim(0, 103); ax.set_ylim(0, 32)
    names = ["PDF\nfiling", "Parse\n+ tables", "Section\nchunks\n+ parents", "Embed +\nBM25\nindex", "Hybrid\nsearch\n(RRF)", "Rerank", "Order +\nparent\ncontext", "Generate\n+ check"]
    bw, gap = 10.6, 2.2
    for k, t in enumerate(names):
        x = k * (bw + gap) + 0.4
        ax.add_patch(FancyBboxPatch((x, 10), bw, 13, boxstyle="round,pad=0.1,rounding_size=1.0", fc="#eef3f6" if k < 4 else "#dff1ec", ec=MUTED, lw=1))
        ax.text(x + bw / 2, 16.5, t, ha="center", va="center", fontsize=7.2)
        if k < len(names) - 1:
            ax.annotate("", xy=(x + bw + gap - 0.1, 16.5), xytext=(x + bw + 0.1, 16.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=0))
    ax.plot([0.4, 4 * (bw + gap) - gap + 0.4], [6, 6], color=GREY, lw=3); ax.text(2 * (bw + gap), 2.5, "built once per filing, stored on disk", ha="center", fontsize=8, color=MUTED)
    ax.plot([4 * (bw + gap) + 0.4, 8 * (bw + gap) - gap + 0.4], [6, 6], color=TEAL, lw=3); ax.text(6 * (bw + gap), 2.5, "run for every question", ha="center", fontsize=8, color=MUTED)
    save(fig, "pipeline")


for f in (retrieval_mode, forest, ceiling, by_type, taxonomy, financebench, pipeline):
    f()
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
