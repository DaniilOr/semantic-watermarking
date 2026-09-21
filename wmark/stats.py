import math

import numpy as np

MIN_EFFECTIVE_N = 8.0
CLIP = 0.5
GATE_MIN = 0.2
GATE_REF = 1.0


def gate_weights(h0, h_min=GATE_MIN, h_ref=GATE_REF):
    return np.clip((h0 - h_min) / max(1e-6, h_ref - h_min), 0.0, 1.0)


def weighted_t(x, w=None):
    x = np.asarray(x, dtype=np.float64)
    if w is None:
        if x.size < 2:
            return None
        sd = float(x.std(ddof=1))
        return float(x.mean() / (sd / math.sqrt(x.size))) if sd > 0 else None
    w = np.asarray(w, dtype=np.float64)
    sw, sw2 = float(w.sum()), float((w * w).sum())
    if sw <= 0 or sw2 <= 0 or (sw * sw / sw2) < MIN_EFFECTIVE_N:
        return None
    m = float((w * x).sum() / sw)
    n_eff = sw * sw / sw2
    var = float((w * (x - m) ** 2).sum() / sw) * n_eff / max(1.0, n_eff - 1.0)
    if var <= 0:
        return None
    return m / (math.sqrt(var) * math.sqrt(sw2) / sw)


def statistic(h0, d, variant="clip_gate"):
    if variant == "plain":
        return weighted_t(d)
    if variant == "clip":
        return weighted_t(np.clip(d, -CLIP, CLIP))
    if variant == "clip_gate":
        return weighted_t(np.clip(d, -CLIP, CLIP), gate_weights(h0))
    raise ValueError(variant)


VARIANTS = ("clip_gate", "clip", "plain")


def upper_tail_threshold(values, alpha):
    a = np.sort(np.asarray([v for v in values if v is not None], dtype=np.float64))
    k = int(math.floor(alpha * (len(a) + 1))) - 1
    return float(a[len(a) - 1 - k]) if k >= 0 and len(a) else float("inf")


def auroc(pos, neg):
    pos = np.asarray([v for v in pos if v is not None], dtype=np.float64)
    neg = np.asarray([v for v in neg if v is not None], dtype=np.float64)
    if not pos.size or not neg.size:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
