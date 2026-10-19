"""Is a watermarked setting's flaw count within what the CLEAN generator produces anyway?

The clean generator is small and flawed, so "the watermarked text has flaws" means nothing by itself.  This script
estimates the clean baseline SEPARATELY, from several independent clean draws of the same prompts (pointwise judge,
scripts/api_pointwise.py), and then asks of every watermarked arm:

  1. band      -- is its rate inside the 95% band of what ONE clean draw of these prompts gives?
                  (band: resample prompts, and for every prompt pick one of the K clean draws at random)
  2. difference -- paired difference to the per-prompt clean expectation (mean over the K clean draws), with a
                  bootstrap interval over prompts; "not worse" if the interval's upper end stays below the margin.

Metrics: share of texts with a major flaw, mean number of flaws per text, share not fully on task, mean 1-5 score.

Usage: python scripts/quality_vs_clean.py --clean runs/tradeoff:base --clean runs/cross_llama:base --clean runs/clean_draws:base \
           --marked runs/tradeoff [--margin 0.10] [--out runs/quality_vs_clean.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "google_gemma-4-31b-it"
METRICS = {
    "major_flaw": (lambda j: 1.0 if j["severity"] == "major" else 0.0, "share of texts with a major flaw", +1),
    "n_flaws": (lambda j: float(len(set(j["flaws"]))), "flaws per text", +1),
    "not_on_task": (lambda j: 0.0 if j["addresses_task"] == "yes" else 1.0, "share not fully on task", +1),
    "insertions_or_drift": (lambda j: 1.0 if {"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"]) else 0.0,
                            "share with unrelated insertions or topic drift", +1),
    "quality": (lambda j: float(j["quality"]), "mean 1-5 score", -1),
}


def load(run: str) -> list[dict]:
    p = Path(run) / "eval" / f"api_pointwise_{SLUG}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def clean_draws(specs: list[str]) -> list[dict[str, dict]]:
    """Each draw: prompt_id -> judgment.  A run with several seeds of the arm contributes one draw per seed."""
    draws = []
    for spec in specs:
        run, arm = spec.split(":")
        by_seed = defaultdict(dict)
        for r in load(run):
            if r["arm"] == arm:
                by_seed[r.get("seed")][r["prompt_id"]] = r["judgment"]
        draws += [d for _, d in sorted(by_seed.items(), key=lambda kv: str(kv[0])) if len(d) >= 30]
    return draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="append", required=True, help="run:arm of a clean draw (repeatable)")
    ap.add_argument("--marked", action="append", required=True, help="run directory whose non-base arms are tested")
    ap.add_argument("--margin", type=float, default=0.10, help="non-inferiority margin for rates (absolute)")
    ap.add_argument("--out", default="runs/quality_vs_clean.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    draws = clean_draws(a.clean)
    prompts = sorted(set.intersection(*[set(d) for d in draws]))
    K, P = len(draws), len(prompts)
    print(f"clean baseline: {K} independent clean draws x {P} shared prompts\n")
    out = {"n_clean_draws": K, "n_prompts": P, "margin": a.margin, "metrics": {}}
    for m, (fn, label, bad_dir) in METRICS.items():
        C = np.array([[fn(d[p]) for p in prompts] for d in draws])              # [K, P]
        per_draw = C.mean(axis=1)
        band = []
        for _ in range(4000):                                                     # what ONE clean draw of P prompts looks like
            idx = rng.integers(0, P, size=P)
            band.append(C[rng.integers(0, K, size=P), idx].mean())
        lo, hi = np.percentile(band, [2.5, 97.5])
        exp = C.mean(axis=0)                                                      # per-prompt clean expectation
        rec = {"label": label, "clean_mean": float(C.mean()), "clean_per_draw": [float(x) for x in per_draw],
               "clean_band95": [float(lo), float(hi)], "arms": {}}
        print(f"== {label} ==\n   clean: mean {C.mean():.3f}; per draw {', '.join(f'{x:.3f}' for x in per_draw)}; 95% band for one clean draw [{lo:.3f}, {hi:.3f}]")
        for run in a.marked:
            by_arm = defaultdict(dict)
            for r in load(run):
                if r["arm"] != "base":
                    by_arm[r["arm"]][r["prompt_id"]] = r["judgment"]
            for arm, d in sorted(by_arm.items()):
                use = [i for i, p in enumerate(prompts) if p in d]
                if len(use) < 30:
                    continue
                w = np.array([fn(d[prompts[i]]) for i in use]); e = exp[use]
                diff = w - e
                boot = [float(diff[rng.integers(0, len(use), size=len(use))].mean()) for _ in range(4000)]
                dlo, dhi = np.percentile(boot, [2.5, 97.5])
                inside = bool(lo <= w.mean() <= hi)
                worse = (dlo > 0) if bad_dir > 0 else (dhi < 0)
                margin = a.margin if m != "quality" and m != "n_flaws" else (0.25 if m == "quality" else 0.25)
                noninferior = (dhi < margin) if bad_dir > 0 else (dlo > -margin)
                rec["arms"][f"{Path(run).name}/{arm}"] = {"n": len(use), "value": float(w.mean()), "inside_clean_band": inside,
                                                         "diff": float(diff.mean()), "diff_ci95": [float(dlo), float(dhi)],
                                                         "significantly_worse": bool(worse), "non_inferior_at_margin": bool(noninferior), "margin": margin}
                tag = "inside clean band" if inside else "OUTSIDE clean band"
                print(f"   {arm:12s} n={len(use):3d} value={w.mean():.3f}  ({tag})  diff vs clean {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                      f"{'  significantly worse' if worse else ''}{'  non-inferior' if noninferior else ''}")
        out["metrics"][m] = rec
        print()
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
