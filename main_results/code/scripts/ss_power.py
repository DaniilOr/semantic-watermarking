"""Power model and alternative segment statistics for the semantic watermark (step 0 of round 2; CPU, no generation).

1. Null anatomy of the segment statistic: per-unit mean / sd of the clipped depth, and the within-text (intraclass)
   correlation rho that a fixed target direction inherits from the text's topic.  Var(t) ~ sigma^2 (1 + (n-1) rho).
2. Alternative statistics re-scored on existing marked runs against the same null:
     t_seg      clipped depth inside the one fixed bin (min over planes), as shipped in semwm.seg_stat
     t_dir      clipped projection on the single target direction u (sum of the bin's signed normals, unit length)
     t_dir_raw  the same, unclipped
     t_seg2     clipped depth inside the nearer of the mask's two bins (max over bins, then min over planes)
3. Predicted TPR@alpha for a per-unit shift, by shifting the null texts' own units (topic structure and clipping kept):
   a uniform shift delta, and best-of-N selection with within-context sd sigma_w (shift = max of N draws - one draw).
   Calibration: the uniform shift measured on round-1 steered arms is fed back in and compared with their observed TPR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.posstats import auroc, upper_tail_threshold, wilson  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

CLIP = 0.05


def texts_of(run: str, arms: set[str] | None, min_tokens: int) -> list[dict]:
    gens, _ = collect_run_texts(Path(run))
    return [g for g in gens if g.get("status", "ok") == "ok" and g.get("text") and int(g.get("visible_tokens") or 0) >= min_tokens
            and (arms is None or g["arm"] in arms)]


def unit_values(seg_emb: np.ndarray, planes: semwm.Hyperplanes, target: int, bins2: tuple[int, ...], u: np.ndarray) -> dict:
    """Per-unit raw values of every statistic (before clipping)."""
    p = planes.projections(seg_emb)                                        # [n, d]
    depth = np.min(planes.bin_signs(target) * p, axis=1)
    depth2 = np.max(np.stack([np.min(planes.bin_signs(b) * p, axis=1) for b in bins2]), axis=0)
    return {"seg": depth, "seg2": depth2, "dir": np.asarray(seg_emb, dtype=np.float64) @ u}


def stat(v: np.ndarray, clip: float | None) -> float:
    x = np.clip(v, -clip, clip) if clip is not None else v
    return float(np.sort(x).sum() / np.sqrt(len(x)))


STATS = {"t_seg": ("seg", CLIP), "t_dir": ("dir", CLIP), "t_dir_raw": ("dir", None), "t_seg2": ("seg2", CLIP),
         "z_seg": ("seg", CLIP), "z_dir": ("dir", CLIP)}
CENTRE: dict[str, tuple[float, float]] = {}      # z_*: per-unit null mean / sd of the clipped value (set from the null)


def stat_named(name: str, v: np.ndarray) -> float:
    """t_*: sum / sqrt(n) as shipped.  z_*: centred on the null's per-unit mean, so the null no longer drifts with n."""
    field, clip = STATS[name]
    x = np.clip(v, -clip, clip) if clip is not None else v
    if name.startswith("z_"):
        mu, sd = CENTRE[field]
        return float((np.sort(x).sum() - len(x) * mu) / (sd * np.sqrt(len(x))))
    return float(np.sort(x).sum() / np.sqrt(len(x)))


def icc(groups: list[np.ndarray]) -> dict:
    """One-way ANOVA intraclass correlation over texts (unequal group sizes)."""
    groups = [g for g in groups if len(g) >= 2]
    k, N = len(groups), sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return {"rho": float((msb - msw) / (msb + (n0 - 1) * msw)), "sd_unit": float(np.concatenate(groups).std()),
            "sd_within": float(np.sqrt(msw)), "n_texts": k, "mean_units": float(N / k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="append", required=True)
    ap.add_argument("--marked", action="append", default=[])
    ap.add_argument("--smoke", default="", help="bestof smoke/production run: sigma_w from depth_chosen - depth_mean")
    ap.add_argument("--embedder", default="/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default="runs/ss_power_segemb.npz")
    ap.add_argument("--out", default="runs/ss_power.json")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    key = semwm.Key.shipped()
    emb = semwm.Embedder(a.embedder, device=a.device, batch_size=256)
    planes = semwm.Hyperplanes.for_key(key, emb.dim)
    target, bins2 = semwm.fixed_bin(key), semwm.valid_mask(key, 0)
    u = semwm.target_direction(planes, planes.bin_signs(target), np.ones(planes.d, dtype=bool))

    pool, groups = {}, defaultdict(list)
    for kind, runs in (("null", a.null), ("marked", a.marked)):
        for run in runs:
            for g in texts_of(run, {"base"} if kind == "null" else None, a.min_tokens):
                pool.setdefault(g["text_hash"], g["text"])
                groups[(kind, run)].append((g["text_hash"], g["arm"]))

    cache = {}
    if Path(a.cache).exists():
        z = np.load(a.cache, allow_pickle=True)
        if int(z["W"]) == a.W and str(z["embedder"]) == a.embedder:
            cache = z["emb"].item()
    todo = [h for h in sorted(pool) if h not in cache]
    print(f"{len(pool)} texts, {len(todo)} to embed ({time.time() - t0:.0f}s)", flush=True)
    for s0 in range(0, len(todo), 256):
        hs = todo[s0:s0 + 256]
        segs = [semwm.split_segments(pool[h], a.W) for h in hs]
        flat = emb.encode([s for ss in segs for s in ss])
        i = 0
        for h, ss in zip(hs, segs):
            cache[h] = flat[i:i + len(ss)]; i += len(ss)
        print(f"  embedded {min(s0 + 256, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    if todo:
        np.savez(a.cache, emb=np.array(cache, dtype=object), W=a.W, embedder=a.embedder)

    uv = {h: unit_values(cache[h], planes, target, bins2, u) for h in pool if len(cache[h])}

    # ---- 1. null anatomy ---- #
    null_h = sorted({h for (kind, _), items in groups.items() if kind == "null" for h, _ in items if h in uv})
    for f in ("seg", "dir"):
        allx = np.clip(np.concatenate([uv[h][f] for h in null_h]), -CLIP, CLIP)
        CENTRE[f] = (float(allx.mean()), float(allx.std()))
    out = {"n_null": len(null_h), "alpha": a.alpha, "W": a.W, "clip": CLIP, "null": {}, "runs": {}, "predict": {}}
    null_vals = {}
    for name, (field, clip) in STATS.items():
        vals = [stat_named(name, uv[h][field]) for h in null_h]
        per_unit = [np.clip(uv[h][field], -clip, clip) if clip else uv[h][field] for h in null_h]
        an = icc(per_unit)
        n_bar = an["mean_units"]
        null_vals[name] = vals
        out["null"][name] = {**an, "unit_mean": float(np.mean(np.concatenate(per_unit))), "thr": upper_tail_threshold(vals, a.alpha),
                             "stat_mean": float(np.mean(vals)), "stat_sd": float(np.std(vals, ddof=1)),
                             "stat_sd_if_independent": an["sd_unit"], "variance_inflation_pred": 1 + (n_bar - 1) * an["rho"],
                             "variance_inflation_obs": float(np.var(vals, ddof=1) / an["sd_unit"] ** 2)}
        for (kind, run), items in groups.items():                          # rho per null source (Qwen vs Llama clean)
            if kind == "null":
                out["null"][name][f"rho[{Path(run).name}]"] = icc([per_unit[null_h.index(h)] for h, _ in items if h in uv])["rho"]
        print(f"null {name:10s} rho={an['rho']:.3f} unit sd={an['sd_unit']:.4f} within sd={an['sd_within']:.4f} "
              f"n={n_bar:.1f} inflation pred {out['null'][name]['variance_inflation_pred']:.2f} obs {out['null'][name]['variance_inflation_obs']:.2f} "
              f"thr={out['null'][name]['thr']:.4f}", flush=True)

    # ---- 2. alternative statistics on marked runs ---- #
    null_unit_mean = {f: float(np.mean(np.concatenate([uv[h][f] for h in null_h]))) for f in ("seg", "dir", "seg2")}
    for (kind, run), items in groups.items():
        if kind != "marked":
            continue
        by_arm = defaultdict(list)
        for h, arm in items:
            if h in uv:
                by_arm[arm].append(h)
        rr = {}
        for arm, hs in sorted(by_arm.items()):
            rec = {"n": len(hs), "unit_shift_raw": {f: float(np.mean(np.concatenate([uv[h][f] for h in hs]))) - null_unit_mean[f]
                                                    for f in ("seg", "dir", "seg2")}}
            line = f"{Path(run).name:26s} {arm:20s} n={len(hs):3d}"
            for name, (field, clip) in STATS.items():
                v = [stat_named(name, uv[h][field]) for h in hs]
                hits = sum(x > out["null"][name]["thr"] for x in v)
                rec[name] = {"tpr": hits / len(v), "ci": wilson(hits, len(v)), "auroc": auroc(v, null_vals[name])}
                line += f" | {name} {rec[name]['tpr']:.2f} ({rec[name]['auroc']:.3f})"
            rr[arm] = rec
            print(line, flush=True)
        out["runs"][Path(run).name] = rr

    # ---- 3. predicted TPR by shifting the null's own units ---- #
    def predict(name: str, shift_fn, reps: int = 4) -> float:
        field = STATS[name][0]
        thr = out["null"][name]["thr"]
        hits = 0
        for _ in range(reps):
            for h in null_h:
                hits += stat_named(name, uv[h][field] + shift_fn(len(uv[h][field]))) > thr
        return hits / (reps * len(null_h))

    grid = {}
    for name, (field, clip) in STATS.items():
        sd = out["null"][name]["sd_unit"]
        grid[name] = {"delta_in_unit_sd": {}, "bestof4_sigma_w_in_unit_sd": {}}
        for k in (0.25, 0.5, 0.75, 1.0, 1.5):
            grid[name]["delta_in_unit_sd"][str(k)] = predict(name, lambda n, d=k * sd: np.full(n, d))
            grid[name]["bestof4_sigma_w_in_unit_sd"][str(k)] = predict(
                name, lambda n, s=k * sd: s * (rng.standard_normal((n, 4)).max(axis=1) - rng.standard_normal(n)))
        print(f"predict {name:10s} uniform delta (unit sd) " + " ".join(f"{k}:{v:.2f}" for k, v in grid[name]["delta_in_unit_sd"].items())
              + " | best-of-4 sigma_w " + " ".join(f"{k}:{v:.2f}" for k, v in grid[name]["bestof4_sigma_w_in_unit_sd"].items()), flush=True)
    out["predict"]["grid"] = grid
    # calibration: each marked arm's own mean unit shift, fed back as a uniform shift
    calib = {}
    for run, rr in out["runs"].items():
        for arm, rec in rr.items():
            calib[f"{run}:{arm}"] = {name: {"observed": rec[name]["tpr"],
                                            "predicted_uniform": predict(name, lambda n, d=rec["unit_shift_raw"][f]: np.full(n, d), reps=1)}
                                     for name, (f, c) in STATS.items()}
    out["predict"]["calibration"] = calib
    for k, v in calib.items():
        print(f"calib {k:45s} " + " ".join(f"{n} obs {x['observed']:.2f} pred {x['predicted_uniform']:.2f}" for n, x in v.items()), flush=True)

    if a.smoke:
        gaps = [(g["cost"]["depth_chosen"] - g["cost"]["depth_mean"]) for g in texts_of(a.smoke, None, 0)
                if (g.get("cost") or {}).get("depth_chosen") is not None]
        if gaps:
            sw = float(np.mean(gaps)) / 1.0294                               # E[max of 4 N(0,1)] - E[N(0,1)]
            sd = out["null"]["t_seg"]["sd_unit"]
            out["predict"]["bestof4_from_smoke"] = {"sigma_w": sw, "sigma_w_in_unit_sd": sw / sd, "n_texts": len(gaps),
                                                    "tpr_t_seg": predict("t_seg", lambda n: sw * (rng.standard_normal((n, 4)).max(axis=1) - rng.standard_normal(n)))}
            print(f"best-of-4 from {a.smoke}: sigma_w={sw:.4f} ({sw / sd:.2f} unit sd) -> predicted t_seg TPR "
                  f"{out['predict']['bestof4_from_smoke']['tpr_t_seg']:.2f}", flush=True)
    write_json(a.out, out)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
