"""Evidence bought per nat of divergence, by push strength (existing per-token traces; first-order theory says T^2 = 2 * total KL)."""
import json
from pathlib import Path

import numpy as np

print(f"{'run/arm':24s} {'beta':>5s} {'tokens':>6s} {'KL/token':>9s} {'KL total':>9s} {'KL first 64 tok':>15s} {'T':>6s} {'T^2/(2 KL)':>11s}")
for run, arm, beta in (("keylen10", "wm_c2", 2), ("strength_fill", "wm_c25", 2.5), ("strength_fill", "wm_c3", 3), ("strength_fill", "wm_c35", 3.5),
                       ("keylen10", "wm_c4", 4), ("long_story", "wm_c2", 2), ("long_story", "wm_c3", 3)):
    T, KL, KL64, N = [], [], [], []
    for f in (Path("runs") / run).glob("trajectories*.jsonl"):
        for ln in open(f):
            if '"per_token"' not in ln or f".{arm}." not in ln[:300]:
                continue
            pt = json.loads(ln)["cost"]["per_token"]
            ev = np.clip(np.asarray(pt["contrast_raw"], dtype=np.float64), -0.5, 0.5)
            kl = np.asarray(pt["kl"], dtype=np.float64)
            if ev.size < 32:
                continue
            T.append(ev.mean() / (ev.std(ddof=1) / np.sqrt(ev.size))); KL.append(kl.sum()); KL64.append(kl[:64].sum()); N.append(ev.size)
    T, KL = np.array(T), np.array(KL)
    print(f"{run + '/' + arm:24s} {beta:5.1f} {int(np.median(N)):6d} {KL.sum() / np.sum(N):9.4f} {KL.mean():9.1f} {np.mean(KL64) / KL.mean():15.0%} {T.mean():6.2f} {np.mean(T) ** 2 / (2 * KL.mean()):11.3f}")
