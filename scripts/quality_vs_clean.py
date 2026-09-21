import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import _path
from wmark.io import read_jsonl, write_json
from wmark.openrouter import DEFAULT_MODEL, slug

METRICS = {
    "major_flaw": (lambda j: 1.0 if j["severity"] == "major" else 0.0, +1, 0.10),
    "n_flaws": (lambda j: float(len(set(j["flaws"]))), +1, 0.25),
    "not_on_task": (lambda j: 0.0 if j["addresses_task"] == "yes" else 1.0, +1, 0.10),
    "insertions_or_drift": (lambda j: 1.0 if {"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"]) else 0.0, +1, 0.10),
    "quality": (lambda j: float(j["quality"]), -1, 0.25),
}


def load(run, model):
    return read_jsonl(Path(run) / f"flaws_{slug(model)}.jsonl")


def clean_draws(specs, model, min_prompts):
    draws = []
    for spec in specs:
        run, arm = spec.split(":")
        by_seed = defaultdict(dict)
        for r in load(run, model):
            if r["arm"] == arm:
                by_seed[r.get("seed")][r["prompt_id"]] = r["judgment"]
        draws += [d for _, d in sorted(by_seed.items(), key=lambda kv: str(kv[0])) if len(d) >= min_prompts]
    return draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="append", required=True, help="run:arm of a clean draw (repeatable, >= 2 draws)")
    ap.add_argument("--marked", action="append", required=True)
    ap.add_argument("--clean-arms", default="base")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-prompts", type=int, default=30)
    ap.add_argument("--out", default="quality_vs_clean.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    draws = clean_draws(a.clean, a.model, a.min_prompts)
    prompts = sorted(set.intersection(*[set(d) for d in draws]))
    K, P = len(draws), len(prompts)
    print(f"clean baseline: {K} clean draws x {P} shared prompts")
    out = {"n_clean_draws": K, "n_prompts": P, "metrics": {}}
    skip = set(a.clean_arms.split(","))
    for m, (fn, bad_dir, margin) in METRICS.items():
        C = np.array([[fn(d[p]) for p in prompts] for d in draws])
        band = [C[rng.integers(0, K, size=P), rng.integers(0, P, size=P)].mean() for _ in range(4000)]
        lo, hi = np.percentile(band, [2.5, 97.5])
        exp = C.mean(axis=0)
        rec = {"clean_mean": float(C.mean()), "clean_per_draw": [float(x) for x in C.mean(axis=1)],
               "clean_band95": [float(lo), float(hi)], "arms": {}}
        print(f"== {m}: clean {C.mean():.3f}, 95% band of one clean draw [{lo:.3f}, {hi:.3f}]")
        for run in a.marked:
            by_arm = defaultdict(dict)
            for r in load(run, a.model):
                if r["arm"] not in skip:
                    by_arm[r["arm"]][r["prompt_id"]] = r["judgment"]
            for arm, d in sorted(by_arm.items()):
                use = [i for i, p in enumerate(prompts) if p in d]
                if len(use) < a.min_prompts:
                    continue
                w = np.array([fn(d[prompts[i]]) for i in use])
                diff = w - exp[use]
                boot = [float(diff[rng.integers(0, len(use), size=len(use))].mean()) for _ in range(4000)]
                dlo, dhi = np.percentile(boot, [2.5, 97.5])
                inside = bool(lo <= w.mean() <= hi)
                worse = bool(dlo > 0) if bad_dir > 0 else bool(dhi < 0)
                noninferior = bool(dhi < margin) if bad_dir > 0 else bool(dlo > -margin)
                rec["arms"][f"{Path(run).name}/{arm}"] = {
                    "n": len(use), "value": float(w.mean()), "inside_clean_band": inside, "diff": float(diff.mean()),
                    "diff_ci95": [float(dlo), float(dhi)], "significantly_worse": worse,
                    "non_inferior_at_margin": noninferior, "margin": margin}
                print(f"   {arm:14s} n={len(use):3d} value={w.mean():.3f} ({'inside' if inside else 'OUTSIDE'} clean band) "
                      f"diff {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]{' worse' if worse else ''}{' non-inferior' if noninferior else ''}")
        out["metrics"][m] = rec
    write_json(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
