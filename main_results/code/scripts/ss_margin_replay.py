"""What does a soft margin cost the bounded chain?  Replay on units that logged all 16 candidates (runs/ss_chain_smoke,
arms chain_*_bo16 and chain_*_bo16_half).  Rule = production rule with a margin m in the STOP test: draw blocks of 4
(cap 16), stop at the first block holding a candidate that is in the mask AND at least m from every hyperplane; keep
the deepest such candidate; at the cap keep the deepest in-mask one, else the closest.  m = 0 is the production rule;
SemStamp uses m = 0.02 with rejection up to 100.  Reported: candidates per unit, units in the mask, units clearing m,
the share of kept units clearing SemStamp's 0.02 (what makes a unit survive rewording), and chain-z detection at the frozen thresholds (generation-side units).  Paraphrase robustness is NOT measured here.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402
from wmworld.records import write_json  # noqa: E402


def replay(units, m, key, planes, lmbd, rng, B=4, cap=16):
    prev = units[0]["bin"]
    inm, clear, drawn = [], [], []
    for u in units[1:]:
        C = np.asarray(u["cands"])[:, :planes.d]
        mask = semwm.valid_mask(key, prev, lmbd)
        order = rng.permutation(len(C))[:cap]
        seen = []
        for s0 in range(0, len(order), B):
            seen += list(order[s0:s0 + B])
            if any(planes.bin_of(C[i]) in mask and np.min(np.abs(C[i])) >= m for i in order[s0:s0 + B]):
                break
        d = {i: semwm.signed_distance(C[i], mask, planes) for i in seen}
        ok = [i for i in seen if d[i] > 0 and np.min(np.abs(C[i])) >= m]
        j = max(ok, key=d.get) if ok else max(seen, key=d.get)
        inm.append(d[j] > 0); clear.append(bool(np.min(np.abs(C[j])) >= semwm.MARGIN) and d[j] > 0); drawn.append(len(seen))
        prev = planes.bin_of(C[j])
    n = len(inm)
    z = (sum(inm) - lmbd * n) / math.sqrt(n * lmbd * (1 - lmbd))
    return z, float(np.mean(inm)), float(np.mean(clear)), float(np.mean(drawn))


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ss_chain_smoke")
    thr = json.loads(Path("runs/ss_budget.json").read_text())["thresholds"]
    key, lmbd = semwm.Key.shipped(), semwm.LMBD
    planes = semwm.Hyperplanes(np.eye(3))                      # only bin_of / signs on logged projections are used
    rng = np.random.default_rng(0)
    by = defaultdict(list)
    for g in collect_run_texts(run)[0]:
        if g["arm"] in ("chain_sent_bo16", "chain_seg_bo16", "chain_sent_bo16_half", "chain_seg_bo16_half") and g.get("status", "ok") == "ok":
            by["seg" if "_seg_" in g["arm"] else "sent"].append(g["cost"]["units"])
    out = {}
    for unit, texts in sorted(by.items()):
        zs = "z_segchain" if unit == "seg" else "z_chain"
        texts = [t for t in texts if len(t) >= 3]
        print(f"\n{unit} units ({len(texts)} texts, all 16 candidates logged); threshold {zs} > {thr[zs]:.3f}")
        print(f"  {'margin':>6s} {'cand/unit':>9s} {'units in mask':>13s} {'kept units clearing 0.02':>24s} {'detected (z)':>12s}")
        out[unit] = {}
        for m in (0.0, 0.005, 0.01, 0.015, 0.02):
            R = [replay(t, m, key, planes, lmbd, rng) for t in texts for _ in range(20)]
            r = {"cand_per_unit": float(np.mean([x[3] for x in R])), "in_mask": float(np.mean([x[1] for x in R])),
                 "clear_margin": float(np.mean([x[2] for x in R])), "det_z": float(np.mean([x[0] > thr[zs] for x in R]))}
            out[unit][str(m)] = r
            print(f"  {m:6.3f} {r['cand_per_unit']:9.2f} {r['in_mask']:13.3f} {r['clear_margin']:24.3f} {r['det_z']:12.2f}", flush=True)
    write_json("runs/ss_margin_replay.json", out)
    print("\nwrote runs/ss_margin_replay.json")


if __name__ == "__main__":
    main()
