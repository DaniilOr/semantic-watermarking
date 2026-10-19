"""Budget curve of the bounded chain (amendment 7): detection vs candidates per unit, replayed from one logged run.

The chain arm with --best-of 16 logs the projections of every candidate of every unit.  For a budget N and a keep
rule we replay the chain: at unit i draw N of its 16 candidates (random order), apply the rule against the mask keyed
by the bin chosen at unit i-1 IN THE REPLAY, and score the text with the detector's chain statistics:
  best      keep the candidate deepest inside the mask (or closest to it)          -- ours
  semstamp  first candidate in the mask AND clearing the 0.02 margin, else the N-th (forced through)
  cap       first candidate in the mask (no margin), else the N-th                  -- capped rejection, k-SemStamp style
Approximation: candidates of unit i were sampled after the logged (N=16) choice of unit i-1, not after the replayed one;
candidates are exact samples whatever the mask, so only the context drift differs.

Also reported: the real texts (N=16 path and the 1B-draft arms) scored by the detector itself, the gen-side vs
detect-side in-mask rate (do generator and detector see the same units?), and compute per unit in 3B-token
equivalents (1B tokens weighted by the parameter ratio), against base (1 candidate) and shipped SemStamp (60).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.posstats import upper_tail_threshold, wilson  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

CLIP = 0.05


def chain_stats(in_mask: list[bool], depth: list[float], lmbd: float) -> tuple[float, float]:
    m = len(in_mask)
    z = (sum(in_mask) - lmbd * m) / math.sqrt(m * lmbd * (1 - lmbd))
    t = float(np.sum(np.clip(depth, -CLIP, CLIP)) / math.sqrt(m))
    return z, t


def replay(units: list[dict], N: int, rule: str, key, planes, lmbd: float, margin: float, rng) -> dict:
    prev = units[0]["bin"]
    inm, dep, drawn = [], [], []
    for u in units[1:]:
        C = np.asarray(u["cands"])
        P = C[:, :planes.d]
        mask = semwm.valid_mask(key, prev, lmbd)
        sub = rng.permutation(len(C))[:N]
        d = np.array([semwm.signed_distance(P[i], mask, planes) for i in sub])
        ins = np.array([planes.bin_of(P[i]) in mask for i in sub])
        if rule == "best":
            j = int(np.argmax(d)); n_used = len(sub)
        else:
            ok = ins & (np.min(np.abs(P[sub]), axis=1) >= margin) if rule == "semstamp" else ins
            hit = np.flatnonzero(ok)
            j = int(hit[0]) if len(hit) else len(sub) - 1
            n_used = j + 1
        inm.append(bool(ins[j])); dep.append(float(d[j])); drawn.append(n_used)
        prev = planes.bin_of(P[sub[j]])
    z, t = chain_stats(inm, dep, lmbd)
    return {"z": z, "t": t, "in_mask": float(np.mean(inm)), "drawn": float(np.mean(drawn))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ss_chain_smoke")
    ap.add_argument("--detect-json", default="runs/detect_ss_llama3b.json", help="frozen sentence-chain thresholds")
    ap.add_argument("--null", action="append", default=["runs/null_calibration", "runs/ss_llama3b_null"])
    ap.add_argument("--segcache", default="runs/ss_power_segemb.npz")
    ap.add_argument("--embedder", default="/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="runs/ss_budget.json")
    a = ap.parse_args()
    key, lmbd = semwm.Key.shipped(), semwm.LMBD
    rng = np.random.default_rng(0)

    # ---- thresholds: sentence chain frozen from the production detector; segment chain from the null's cached units ---- #
    thr = dict(json.loads(Path(a.detect_json).read_text())["thresholds"])
    z = np.load(a.segcache, allow_pickle=True)
    cache = z["emb"].item()
    dim = next(iter(cache.values())).shape[1]
    planes = semwm.Hyperplanes.for_key(key, dim)
    null_h = sorted({g["text_hash"] for r in a.null for g in collect_run_texts(Path(r))[0] if g["arm"] == "base" and len(cache.get(g["text_hash"], [])) >= 2})
    segz, segt = [], []
    for h in null_h:
        pg = planes.projections(cache[h])
        segz.append(semwm.chain_z([planes.bin_of(p) for p in pg], key, lmbd)); segt.append(semwm.cont_stat(pg, key, planes, lmbd, CLIP))
    thr["z_segchain"], thr["t_segcont"] = upper_tail_threshold(segz, 0.01), upper_tail_threshold(segt, 0.01)
    print("thresholds (1% FPR): " + " ".join(f"{k}={v:.4f}" for k, v in thr.items()) + f"  (segment chain from {len(null_h)} null texts)", flush=True)
    stat_of = {"sent": ("z_chain", "t_cont"), "seg": ("z_segchain", "t_segcont")}

    gens = [g for g in collect_run_texts(Path(a.run))[0] if g.get("status", "ok") == "ok" and g.get("text")]
    by_arm = defaultdict(list)
    for g in gens:
        by_arm[g["arm"]].append(g)
    base_tok = float(np.mean([g["visible_tokens"] for g in by_arm.get("base", [])])) if by_arm.get("base") else None
    emb = semwm.Embedder(a.embedder, device="cpu", batch_size=128)
    out = {"thresholds": thr, "arms": {}, "curves": {}}

    print(f"\n{'arm':26s} {'n':>3s} {'units':>5s} {'gen in-mask':>11s} {'det in-mask':>11s} {'3B-eq tok/unit':>14s} {'x base':>6s} "
          f"{'z det':>12s} {'t det':>12s} {'s/text':>6s}")
    for arm, gs in sorted(by_arm.items()):
        if arm == "base":
            continue
        unit = "seg" if "_seg_" in arm else "sent"
        zs, ts = stat_of[unit]
        det, gen_rate, det_rate, cost_u = [], [], [], []
        for g in gs:
            s = semwm.score_text(g["text"], key, emb, W=a.W, planes=planes)
            det.append((s[zs], s[ts]))
            c = g["cost"]
            gen_rate.append(c.get("in_mask_rate") or 0.0)
            if unit == "sent":
                det_rate.append(float(np.mean(s["in_mask"])) if s["in_mask"] else 0.0)
            else:
                pg = planes.projections(emb.encode(semwm.split_segments(g["text"], a.W)))
                b = [planes.bin_of(p) for p in pg]
                det_rate.append(float(np.mean([b[i] in semwm.valid_mask(key, b[i - 1], lmbd) for i in range(1, len(b))])) if len(b) > 1 else 0.0)
            cost_u.append((c["t_gen"] + c["t_score"] + c.get("param_ratio", 0) * c["d_gen"]) / max(1, c["n_units"]))
        zv = [x for x, _ in det if x is not None]; tv = [y for _, y in det if y is not None]
        hz, ht = sum(x > thr[zs] for x in zv), sum(y > thr[ts] for y in tv)
        unit_tok = float(np.mean([g["visible_tokens"] / max(1, g["cost"]["n_units"]) for g in gs]))
        rec = {"n": len(gs), "unit": unit, "gen_in_mask": float(np.mean(gen_rate)), "det_in_mask": float(np.mean(det_rate)),
               "cost_3b_tok_per_unit": float(np.mean(cost_u)), "cost_x_base": float(np.mean(cost_u)) / unit_tok,
               "det_z": hz / len(zv), "det_z_ci": wilson(hz, len(zv)), "det_t": ht / len(tv), "det_t_ci": wilson(ht, len(tv)),
               "wall_s": float(np.mean([g["cost"]["wall_seconds"] for g in gs])), "repairs_per_unit": float(np.mean([g["cost"]["repairs"] / max(1, g["cost"]["n_units"]) for g in gs])),
               "fallback_per_unit": float(np.mean([g["cost"]["fallback"] / max(1, g["cost"]["n_units"]) for g in gs]))}
        out["arms"][arm] = rec
        print(f"{arm:26s} {len(gs):3d} {np.mean([g['cost']['n_units'] for g in gs]):5.1f} {rec['gen_in_mask']:11.2f} {rec['det_in_mask']:11.2f} "
              f"{rec['cost_3b_tok_per_unit']:14.1f} {rec['cost_x_base']:6.1f} {rec['det_z']:5.2f} [{rec['det_z_ci'][0]:.2f},{rec['det_z_ci'][1]:.2f}] "
              f"{rec['det_t']:5.2f} [{rec['det_t_ci'][0]:.2f},{rec['det_t_ci'][1]:.2f}] {rec['wall_s']:6.0f}", flush=True)

        # ---- replayed budget curve (3B arms that logged every candidate) ---- #
        if "_d1b" in arm:
            continue
        units_per_text = [g["cost"]["units"] for g in gs if len(g["cost"]["units"]) >= 3]
        curve = {}
        for rule in ("best", "cap", "semstamp"):
            for N in (1, 2, 4, 8, 12, 16):
                R = [replay(u, N, rule, key, planes, lmbd, semwm.MARGIN, rng) for u in units_per_text for _ in range(a.reps)]
                curve[f"{rule}:{N}"] = {"det_z_gen": float(np.mean([r["z"] > thr[zs] for r in R])), "det_t_gen": float(np.mean([r["t"] > thr[ts] for r in R])),
                                        "in_mask": float(np.mean([r["in_mask"] for r in R])), "cand_per_unit": float(np.mean([r["drawn"] for r in R]))}
        out["curves"][arm] = curve
        print(f"  replay (gen-side units; detection at the frozen thresholds; cand/unit = candidates actually drawn):")
        for rule in ("best", "cap", "semstamp"):
            print(f"    {rule:9s} " + " | ".join(f"N={N}: {curve[f'{rule}:{N}']['cand_per_unit']:.1f}c in {curve[f'{rule}:{N}']['in_mask']:.2f} "
                                             f"z {curve[f'{rule}:{N}']['det_z_gen']:.2f} t {curve[f'{rule}:{N}']['det_t_gen']:.2f}" for N in (1, 2, 4, 8, 16)), flush=True)
    out["reference"] = {"semstamp_shipped": {"cand_per_sentence": 60.2, "det_z": 0.68, "wall_s": 320}, "base_tokens": base_tok}
    write_json(a.out, out)
    print(f"\nreference: shipped SemStamp 60.2 candidates/sentence, 68% (z_chain), 320 s/text\nwrote {a.out}")


if __name__ == "__main__":
    main()
