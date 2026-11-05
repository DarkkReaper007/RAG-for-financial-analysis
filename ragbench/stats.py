"""Paired statistics for comparing two configurations on the same questions."""
import math
import random
from statistics import mean


def mean_ci(xs, n_boot=5000, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    m = mean(xs)
    if len(xs) < 5:
        return m, None, None
    rng = random.Random(seed)
    b = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return m, b[int(0.025 * n_boot)], b[int(0.975 * n_boot) - 1]


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Mean of (a - b) over questions with both values, with a 95% bootstrap CI. a, b: equal-length lists (None = missing)."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 5:
        return {"n": len(d), "diff": mean(d) if d else None, "lo": None, "hi": None}
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return {"n": len(d), "diff": mean(d), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1]}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p for paired binary outcomes. Returns (p, a_only, b_only)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n10 = sum(1 for x, y in pairs if x >= 0.5 > y)
    n01 = sum(1 for x, y in pairs if y >= 0.5 > x)
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), n10, n01


def holm(pvals):
    """Holm-Bonferroni adjusted p-values in the original order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj
