"""Per-arm generation cost and mechanics from the records (wall time, length, steering stats)."""
import json, glob, sys
from collections import defaultdict
import numpy as np
run = sys.argv[1] if len(sys.argv) > 1 else "runs/ss_llama3b"
by = defaultdict(list)
for f in glob.glob(f"{run}/generations.*.jsonl"):
    for ln in open(f):
        g = json.loads(ln); by[g["arm"]].append(g)
print(f"{'arm':20s} {'n':>4s} {'s/text':>6s} {'med tok':>7s} {'hit cap':>7s} {'|bias|':>6s} {'Pb before/after':>15s} {'rep hits':>8s}")
for arm, gs in sorted(by.items()):
    c = [g["cost"] for g in gs]
    m = lambda k: np.mean([x.get(k) or 0 for x in c])
    print(f"{arm:20s} {len(gs):4d} {m('wall_seconds'):6.1f} {np.median([g['visible_tokens'] for g in gs]):7.0f} "
          f"{np.mean([g['visible_tokens'] >= 512 for g in gs]):7.2f} {m('steer_bias_abs_mean'):6.2f} "
          f"{m('steer_p_boundary_before'):7.3f}/{m('steer_p_boundary_after'):.3f} {m('steer_rep_guard_hits'):8.1f}")
# chain arms: candidates drawn per unit (the budget SemStamp spends 60.2 of per sentence)
print()
for arm, gs in sorted(by.items()):
    if not arm.startswith("chain_"):
        continue
    U = [u for g in gs for u in g["cost"].get("units", []) if u.get("prev_bin") is not None]
    if not U or "n_drawn" not in U[0]:
        continue
    nd = np.array([u["n_drawn"] for u in U])
    print(f"{arm:20s} units {len(U):5d} | candidates/unit mean {nd.mean():.2f} median {np.median(nd):.0f} | share of units: 4 cand {np.mean(nd == 4):.2f}, 8 {np.mean(nd == 8):.2f}, 12 {np.mean(nd == 12):.2f}, 16 {np.mean(nd == 16):.2f} | gen-side in-mask {np.mean([u['in_mask'] for u in U]):.3f}")
