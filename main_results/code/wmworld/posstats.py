"""Detector variants over per-position statistics (all text-only: no prompt, no generation trace).

The generator pushes only where it is undecided (the entropy gate), and the push moves the realized contrast
``d_t`` by roughly ``beta_t * V_t``, where ``V_t`` is the variance of the contrast over the next-token distribution.
The plain detector averages ``d_t`` over *every* position, including the many that were never touched and the
ones where keyed and neutral worlds agree (no capacity).  A text-only detector can do better, because the
reference model's own entropy ``h0_t`` and the contrast's null mean/variance ``mu_t``/``var_t`` are functions of the
text and the key alone:

* weight positions by an entropy gate (where the generator was probably free to choose),
* centre each position by what an unsteered token would have given (``d_t - mu_t``),
* normalise by the capacity actually available (``sqrt(sum w^2 var)``): the matched filter for this push.

Every variant is calibrated empirically against the same frozen clean null, so none of the modelling
approximations can inflate false positives; they can only cost power.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

MIN_EFFECTIVE_N = 8.0


def gate_weights(h0: np.ndarray, h_min: float, h_ref: float) -> np.ndarray:
    """The generator's entropy ramp, evaluated on the reference model's neutral entropy (0 below h_min, 1 above h_ref)."""
    return np.clip((h0 - h_min) / max(1e-6, h_ref - h_min), 0.0, 1.0)


def weighted_t(x: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """Weighted within-text t-statistic; with unit weights this is the plain detector T = mean/(sd/sqrt(n))."""
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


def matched_sum(d: np.ndarray, mu: np.ndarray, var: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """sum w (d - mu) / sqrt(sum w^2 var): centred evidence over the capacity that was available (known-variance form)."""
    d, mu, var = (np.asarray(a, dtype=np.float64) for a in (d, mu, var))
    w = np.ones_like(d) if w is None else np.asarray(w, dtype=np.float64)
    den = float((w * w * var).sum())
    if den <= 1e-12 or float((w > 0).sum()) < MIN_EFFECTIVE_N:
        return None
    return float((w * (d - mu)).sum() / math.sqrt(den))


def z_sum(d: np.ndarray, mu: np.ndarray, var: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-3) -> float | None:
    """sum w z / sqrt(sum w^2) with z = (d - mu)/sqrt(var + eps): every position standardised on its own scale."""
    d, mu, var = (np.asarray(a, dtype=np.float64) for a in (d, mu, var))
    w = np.ones_like(d) if w is None else np.asarray(w, dtype=np.float64)
    sw2 = float((w * w).sum())
    if sw2 <= 0 or float((w > 0).sum()) < MIN_EFFECTIVE_N:
        return None
    return float((w * (d - mu) / np.sqrt(var + eps)).sum() / math.sqrt(sw2))


def _variants() -> dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float | None]]:
    g_lo = lambda h: gate_weights(h, 0.2, 1.0)        # the generator's own gate settings
    g_hi = lambda h: gate_weights(h, 1.0, 3.0)        # the reference sees no task, so its entropy runs higher
    return {
        "T": lambda h, d, mu, var: weighted_t(d),                                   # the current detector
        "T_gate": lambda h, d, mu, var: weighted_t(d, g_lo(h)),
        "T_gate_hi": lambda h, d, mu, var: weighted_t(d, g_hi(h)),
        "T_centered": lambda h, d, mu, var: weighted_t(d - mu),
        "T_centered_gate": lambda h, d, mu, var: weighted_t(d - mu, g_lo(h)),
        "matched": lambda h, d, mu, var: matched_sum(d, mu, var),
        "matched_gate": lambda h, d, mu, var: matched_sum(d, mu, var, g_lo(h)),
        "matched_gate_hi": lambda h, d, mu, var: matched_sum(d, mu, var, g_hi(h)),
        "z_gate": lambda h, d, mu, var: z_sum(d, mu, var, g_lo(h)),
        # clipped evidence: no single token can matter, so the statistic must be carried by many small preferences.
        # (In the unclipped T the top 5% of tokens carry ~90% of a text's evidence: openers and key-flavoured words.)
        "T_clip05": lambda h, d, mu, var: weighted_t(np.clip(d, -0.5, 0.5)),
        "T_clip1": lambda h, d, mu, var: weighted_t(np.clip(d, -1.0, 1.0)),
        "T_clip05_gate": lambda h, d, mu, var: weighted_t(np.clip(d, -0.5, 0.5), g_lo(h)),
        "T_sign": lambda h, d, mu, var: weighted_t(np.sign(d) * (np.abs(d) > 1e-4)),   # extreme case: only the direction counts
    }


VARIANTS = _variants()


def upper_tail_threshold(values, alpha: float) -> float:
    """Conservative rank threshold: #{null >= threshold} / m <= alpha (infinite when the tail cannot be resolved)."""
    a = np.sort(np.asarray([v for v in values if v is not None], dtype=np.float64))
    k = int(math.floor(alpha * (len(a) + 1))) - 1
    return float(a[len(a) - 1 - k]) if k >= 0 and len(a) else float("inf")


def auroc(pos, neg) -> float | None:
    pos = np.asarray([v for v in pos if v is not None], dtype=np.float64)
    neg = np.asarray([v for v in neg if v is not None], dtype=np.float64)
    if not pos.size or not neg.size:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < allv.size:                                  # midranks for ties
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def write_dump(dest, key_ids: list[str], key_prefixes: list[str], gens: list[dict], stats: list[dict], reference: str = "") -> None:
    """Save per-position statistics of ``gens`` (generation records) as flat arrays with offsets."""
    offsets = np.zeros(len(gens) + 1, dtype=np.int64)
    for j, st in enumerate(stats):
        offsets[j + 1] = offsets[j] + st["h0"].shape[0]
    np.savez_compressed(
        dest, key_ids=np.array(key_ids), key_prefixes=np.array(key_prefixes), offsets=offsets,
        h0=np.concatenate([st["h0"] for st in stats]), d=np.concatenate([st["d"] for st in stats], axis=1),
        mu=np.concatenate([st["mu"] for st in stats], axis=1), var=np.concatenate([st["var"] for st in stats], axis=1),
        text_hash=np.array([g["text_hash"] for g in gens]), arm=np.array([g["arm"] for g in gens]),
        assigned_key_id=np.array([str(g.get("assigned_key_id") or "") for g in gens]),
        gen_id=np.array([str(g["id"]) for g in gens]), domain=np.array([str(g.get("domain") or "") for g in gens]),
        n_gen_tokens=np.array([int(g["visible_tokens"]) for g in gens], dtype=np.int64), reference=np.array(reference))


class Dump:
    """A saved per-position dump: flat arrays plus offsets (see scripts/posstats_dump.py)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.key_ids = [str(x) for x in z["key_ids"]]
        self.kidx = {k: i for i, k in enumerate(self.key_ids)}
        self.offsets = z["offsets"]
        self.h0, self.d, self.mu, self.var = z["h0"], z["d"], z["mu"], z["var"]
        self.text_hash = [str(x) for x in z["text_hash"]]
        self.arm = [str(x) for x in z["arm"]]
        self.assigned = [str(x) for x in z["assigned_key_id"]]
        self.gen_id = [str(x) for x in z["gen_id"]]
        self.n_gen_tokens = z["n_gen_tokens"]

    def __len__(self) -> int:
        return len(self.text_hash)

    def n_tokens(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def stat(self, variant: str, i: int, key_id: str, lo: int = 0, hi: int | None = None) -> float | None:
        """Variant statistic of text i under a key, optionally on the token window [lo, hi) of the text."""
        k = self.kidx.get(key_id)
        if k is None:
            return None
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        s, e = a + lo, (b if hi is None else min(b, a + hi))
        if e - s < 2:
            return None
        return VARIANTS[variant](self.h0[s:e], self.d[k, s:e], self.mu[k, s:e], self.var[k, s:e])
