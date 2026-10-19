"""Trajectory diagnostics for a run: clip saturation and selection decisiveness per length bin (no model needed)."""
import json, glob, sys, collections, statistics
run = sys.argv[1] if len(sys.argv) > 1 else "runs/dev"
gens = {}
for f in sorted(glob.glob(f"{run}/generations*.jsonl")):
    for l in open(f):
        g = json.loads(l); gens[g["id"]] = g
trs = []
for f in sorted(glob.glob(f"{run}/trajectories*.jsonl")):
    for l in open(f):
        t = json.loads(l); t["arm"] = gens[t["output_id"]]["arm"]; trs.append(t)
for arm in sorted({t["arm"] for t in trs}):
    T = [t for t in trs if t["arm"] == arm and t["scale_status"] in ("ok", "rank")]
    if not T:
        continue
    print(f"\n{arm}: {len(T)} active decisions over {len({t['output_id'] for t in T})} outputs")
    bybin = collections.defaultdict(list)
    for t in T:
        if t["sigma"]:
            z = [g / t["sigma"] for g in t["gains"] if g is not None]
            full = all(abs(x) > 5 for x in z); allneg = all(x < -5 for x in z); allpos = all(x > 5 for x in z); spread = max(z) - min(z)
        else:
            full = allneg = allpos = False; spread = float("nan")
        bybin[t["length_bin"]].append((full, allneg, allpos, max(t["probabilities"]), spread, t["gains"][t["chosen"]]))
    for b in ("[0,64)", "[64,128)", "[128,256)", "[256,inf)"):
        v = bybin.get(b, [])
        if not v:
            continue
        print(f"  bin {b:10s} n={len(v):5d} fully-clipped={statistics.mean(x[0] for x in v):.3f} (all<-5: {statistics.mean(x[1] for x in v):.3f}, "
              f"all>5: {statistics.mean(x[2] for x in v):.3f}) median max P={statistics.median(x[3] for x in v):.2f} "
              f"median z-spread={statistics.median(x[4] for x in v):.1f} mean chosen gain={statistics.mean(x[5] for x in v if x[5] is not None):+.4f}")
fg = [g["final_score_trace"] for g in gens.values() if g.get("final_score_trace") is not None]
if fg:
    print(f"\nfinal F from generator trace (global arms): mean {statistics.mean(fg):+.4f} sd {statistics.stdev(fg):.4f} min {min(fg):+.3f} max {max(fg):+.3f} n={len(fg)}")
