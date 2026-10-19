"""Early-abort replay for the bounded chain (amendment 7 side experiment; CPU, no generation).

Input: chain arms logged with --log-half (runs/ss_chain_smoke, arms *_half): for every candidate of every unit, the
projections of the whole unit and of its first half (W/2 words for segments, half the tokens for sentences).
Replayed rule (the production rule plus an abort): candidates in random order, blocks of B (cap 16); a candidate whose
half-unit depth inside the mask is below tau is aborted and costs only its first half; the block stops the unit when
a surviving candidate is in the mask; keep the deepest survivor; if nothing survives by the cap, the candidate with the
best half-depth is completed.  Cost = generated tokens (ideal: an aborted row stops consuming compute, which needs a
two-phase generator -- half-units for the block, then only the survivors continue).
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
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

CLIP = 0.05


def replay(units, tau, B, cap, key, planes, lmbd, rng):
    prev = units[0]["bin"]
    inm, dep, drawn, tok, tok_full = [], [], [], [], []
    for u in units[1:]:
        C, H = np.asarray(u["cands"]), np.asarray(u["half"])
        mask = semwm.valid_mask(key, prev, lmbd)
        order = rng.permutation(len(C))[:cap]
        cost = 0.0; surv = []; used = 0; best_half = None
        for s0 in range(0, len(order), B):
            blk = order[s0:s0 + B]; used += len(blk)
            for i in blk:
                hd = semwm.signed_distance(H[i, :planes.d], mask, planes)
                if hd < tau:
                    cost += H[i, planes.d]
                    if best_half is None or hd > best_half[0]:
                        best_half = (hd, i)
                else:
                    cost += C[i, planes.d]; surv.append(i)
            if any(planes.bin_of(C[i, :planes.d]) in mask for i in surv):
                break
        if not surv:                                           # complete the most promising aborted one
            i = best_half[1]; cost += C[i, planes.d] - H[i, planes.d]; surv = [i]
        d = [semwm.signed_distance(C[i, :planes.d], mask, planes) for i in surv]
        j = surv[int(np.argmax(d))]
        inm.append(planes.bin_of(C[j, :planes.d]) in mask); dep.append(max(d)); drawn.append(used); tok.append(cost)
        tok_full.append(float(np.mean(C[:, planes.d])))
        prev = planes.bin_of(C[j, :planes.d])
    m = len(inm)
    z = (sum(inm) - lmbd * m) / math.sqrt(m * lmbd * (1 - lmbd))
    t = float(np.sum(np.clip(dep, -CLIP, CLIP)) / math.sqrt(m))
    return z, t, float(np.mean(inm)), float(np.mean(drawn)), float(np.mean(tok)) / float(np.mean(tok_full))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ss_chain_smoke")
    ap.add_argument("--budget-json", default="runs/ss_budget.json", help="thresholds (sentence and segment chain)")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="runs/ss_abort.json")
    a = ap.parse_args()
    key, lmbd = semwm.Key.shipped(), semwm.LMBD
    thr = json.loads(Path(a.budget_json).read_text())["thresholds"]
    rng = np.random.default_rng(0)
    by_arm = defaultdict(list)
    for g in collect_run_texts(Path(a.run))[0]:
        if g["arm"].endswith("_half") and g.get("status", "ok") == "ok":
            by_arm[g["arm"]].append(g)
    out = {}
    for arm, gs in sorted(by_arm.items()):
        unit = "seg" if "_seg_" in arm else "sent"
        zs, ts = ("z_segchain", "t_segcont") if unit == "seg" else ("z_chain", "t_cont")
        texts = [g["cost"]["units"] for g in gs if len(g["cost"]["units"]) >= 3]
        dim_d = 3
        planes = semwm.Hyperplanes.for_key(key, 768)
        # how well does the half predict the whole?
        hd, fd, fin = [], [], []
        for units in texts:
            for u in units[1:]:
                mask = semwm.valid_mask(key, u["prev_bin"], lmbd)
                for c, h in zip(u["cands"], u["half"]):
                    hd.append(semwm.signed_distance(np.array(h[:dim_d]), mask, planes)); fd.append(semwm.signed_distance(np.array(c[:dim_d]), mask, planes))
                    fin.append(planes.bin_of(np.array(c[:dim_d])) in mask)
        hd, fd, fin = map(np.asarray, (hd, fd, fin))
        rec = {"n_texts": len(texts), "corr_half_full_depth": float(np.corrcoef(hd, fd)[0, 1]), "p_in_mask": float(fin.mean()), "taus": {}}
        print(f"\n{arm} ({len(texts)} texts, {len(hd)} candidates): corr(half depth, full depth) = {rec['corr_half_full_depth']:.2f}; "
              f"P(in mask) = {fin.mean():.2f}")
        print(f"  {'tau':>7s} {'P(in|abort)':>11s} {'aborted':>7s} {'cand/unit':>9s} {'tok cost vs 1 cand':>18s} {'in-mask':>7s} {'det z':>6s} {'det t':>6s}")
        for tau in (-1.0, -0.06, -0.04, -0.03, -0.02, -0.01, 0.0):
            R = [replay(u, tau, a.block, a.cap, key, planes, lmbd, rng) for u in texts for _ in range(a.reps)]
            ab = hd < tau
            r = {"p_in_given_abort": float(fin[ab].mean()) if ab.any() else None, "abort_rate": float(ab.mean()),
                 "cand_per_unit": float(np.mean([x[3] for x in R])), "tok_cost_rel": float(np.mean([x[4] for x in R])),
                 "in_mask": float(np.mean([x[2] for x in R])), "det_z": float(np.mean([x[0] > thr[zs] for x in R])),
                 "det_t": float(np.mean([x[1] > thr[ts] for x in R]))}
            rec["taus"][str(tau)] = r
            print(f"  {('none' if tau == -1.0 else f'{tau:+.2f}'):>7s} {(r['p_in_given_abort'] if r['p_in_given_abort'] is not None else float('nan')):11.2f} "
                  f"{r['abort_rate']:7.2f} {r['cand_per_unit']:9.1f} {r['tok_cost_rel']:18.2f} {r['in_mask']:7.2f} {r['det_z']:6.2f} {r['det_t']:6.2f}", flush=True)
        out[arm] = rec
    write_json(a.out, out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
