"""Offline prototype of a self-keyed segment statistic (no generation; reuses the unit embeddings of scripts/ss_power.py).

Problem (runs/ss_power.json): with ONE fixed target bin, units of a text share its topic, so their depths correlate
(rho ~ 0.10) and the null of the sum is ~4x wider than for independent units; this caps detection at any length.
Design: ``d_c`` extra keyed "cell" planes (independent of the 3 target planes) cut embedding space into 2^d_c cells;
cell c has its own target bin T(c) (keyed).  A unit is scored by its depth inside T(cell(unit)).  Units of one text
spread over cells (within-text sd ~ total sd), so the topic offset along any one target cancels, while the unit's
target is a function of the unit alone (order-free; a paraphrase keeps it iff it keeps the cell).  d_c = 0 is the
shipped fixed-bin statistic.

Reported per d_c: null rho, variance inflation, 1% threshold, predicted TPR for a uniform per-unit shift and for
best-of-4 selection (shift the null's own units, as in ss_power.py), and a robustness proxy: for attacked clean texts,
the share of units whose cell equals the cell of their most similar unit in the original text.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.posstats import upper_tail_threshold  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

CLIP = 0.05
CELL_SEED = 777_001          # key material of the cell planes (independent of the target planes' RandomState(1234))


def icc(groups):
    groups = [g for g in groups if len(g) >= 2]
    k, N = len(groups), sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)), float(np.concatenate(groups).std())


class SelfKey:
    def __init__(self, dim: int, d_c: int, key: semwm.Key):
        self.planes = semwm.Hyperplanes.for_key(key, dim)
        self.d_c = d_c
        self.cells = semwm.Hyperplanes(np.random.RandomState(CELL_SEED).randn(d_c, dim)) if d_c else None
        n_bins = 2 ** key.lsh_dim
        base = semwm.fixed_bin(key)
        rng = np.random.default_rng(CELL_SEED + d_c)
        self.target = [base] if not d_c else [int(x) for x in rng.integers(0, n_bins, 2 ** d_c)]
        self.signs = np.stack([self.planes.bin_signs(b) for b in range(n_bins)])

    def cell(self, e: np.ndarray) -> np.ndarray:
        if not self.d_c:
            return np.zeros(len(e), dtype=int)
        p = self.cells.projections(e) > 0
        return (p * (1 << np.arange(self.d_c - 1, -1, -1))).sum(axis=1)

    def depth(self, e: np.ndarray) -> np.ndarray:
        t = np.asarray(self.target)[self.cell(e)]
        return np.min(self.signs[t] * self.planes.projections(e), axis=1)


def stat(v, clip=CLIP):
    return float(np.sort(np.clip(v, -clip, clip)).sum() / np.sqrt(len(v)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/ss_power_segemb.npz")
    ap.add_argument("--null", action="append", default=["runs/null_calibration", "runs/ss_llama3b_null"])
    ap.add_argument("--orig", default="runs/ss_llama3b")
    ap.add_argument("--attacks", nargs="+", default=["runs/ss_llama3b_reorder_sent", "runs/ss_llama3b_resegment",
                                                     "runs/ss_llama3b_para_light", "runs/ss_llama3b_para_strong"])
    ap.add_argument("--dc", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="runs/ss_selfkey.json")
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=True)
    cache = z["emb"].item()
    dim = next(iter(cache.values())).shape[1]
    rng = np.random.default_rng(0)

    null_h = sorted({g["text_hash"] for r in a.null for g in collect_run_texts(Path(r))[0]
                     if g["arm"] == "base" and g.get("text_hash") in cache and len(cache[g["text_hash"]])})
    orig = {g["id"]: g["text_hash"] for g in collect_run_texts(Path(a.orig))[0] if g["arm"] in ("base", "clean_seed1")}
    pairs = defaultdict(list)
    for r in a.attacks:
        for g in collect_run_texts(Path(r))[0]:
            if g.get("arm") in ("base", "clean_seed1") and g.get("parent_id") in orig:
                h0, h1 = orig[g["parent_id"]], g.get("text_hash")
                if h0 in cache and h1 in cache and len(cache[h0]) and len(cache[h1]):
                    pairs[Path(r).name].append((h0, h1))
    print(f"null texts {len(null_h)}; attack pairs " + ", ".join(f"{k} {len(v)}" for k, v in pairs.items()), flush=True)

    out = {"n_null": len(null_h), "clip": CLIP, "cell_seed": CELL_SEED, "per_dc": {}}
    key = semwm.Key.shipped()
    for d_c in a.dc:
        sk = SelfKey(dim, d_c, key)
        units = [sk.depth(cache[h]) for h in null_h]
        rho, sd = icc([np.clip(u, -CLIP, CLIP) for u in units])
        vals = [stat(u) for u in units]
        thr = upper_tail_threshold(vals, 0.01)
        infl = float(np.var(vals, ddof=1) / np.var(np.clip(np.concatenate(units), -CLIP, CLIP)))
        cells_per_text = float(np.mean([len(set(sk.cell(cache[h]).tolist())) for h in null_h]))

        def predict(shift_fn, reps=4):
            return float(np.mean([stat(u + shift_fn(len(u))) > thr for _ in range(reps) for u in units]))

        grid = {str(k): predict(lambda n, d=k * sd: np.full(n, d)) for k in (0.5, 1.0, 1.5)}
        bo4 = {str(k): predict(lambda n, s=k * sd: s * (rng.standard_normal((n, 4)).max(axis=1) - rng.standard_normal(n)))
               for k in (0.5, 0.9, 1.2)}
        keep = {}
        for att, pr in pairs.items():                         # does a unit keep its cell after the attack?
            agree = []
            for h0, h1 in pr:
                e0, e1 = cache[h0], cache[h1]
                nn = np.argmax(e1 @ e0.T, axis=1)
                agree.extend((sk.cell(e1) == sk.cell(e0)[nn]).tolist())
            keep[att] = float(np.mean(agree))
        out["per_dc"][str(d_c)] = {"rho": rho, "unit_sd": sd, "variance_inflation": infl, "thr": thr, "cells_per_text": cells_per_text,
                                   "pred_uniform_shift_unit_sd": grid, "pred_bestof4_sigma_w_unit_sd": bo4, "cell_kept_under_attack": keep}
        print(f"d_c={d_c}: rho={rho:.3f} inflation={infl:.2f} thr={thr:.4f} cells/text={cells_per_text:.1f} | uniform "
              + " ".join(f"{k}:{v:.2f}" for k, v in grid.items()) + " | best-of-4 " + " ".join(f"{k}:{v:.2f}" for k, v in bo4.items())
              + " | cell kept " + " ".join(f"{k.replace('ss_llama3b_', '')}:{v:.2f}" for k, v in keep.items()), flush=True)
    write_json(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
