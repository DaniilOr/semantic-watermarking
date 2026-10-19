"""Per-bin incremental contrast added by selection vs the candidate spread available (dev trajectories)."""
import json, glob, sys, collections, statistics
run = sys.argv[1] if len(sys.argv) > 1 else "runs/dev"
gens = {json.loads(l)["id"]: json.loads(l) for f in sorted(glob.glob(f"{run}/generations*.jsonl")) for l in open(f)}
by = collections.defaultdict(list)
for f in sorted(glob.glob(f"{run}/trajectories*.jsonl")):
    for l in open(f):
        t = json.loads(l)
        if gens[t["output_id"]]["arm"] != "llm_global" or t["scale_status"] != "ok":
            continue
        c = [x for x in t["incremental_contrast"] if x is not None]
        if len(c) < 2:
            continue
        chosen = t["incremental_contrast"][t["chosen"]]
        by[t["length_bin"]].append((chosen, statistics.mean(c), statistics.stdev(c), max(c), sum(1 for x in c if x > 0) / len(c)))
print("llm_global, per 16-token unit, nats of (keyed - neutral) log-likelihood added by the chosen candidate:")
for b in ("[0,64)", "[64,128)", "[128,256)", "[256,inf)"):
    v = by.get(b, [])
    if not v: continue
    print(f"  bin {b:10s} n={len(v):5d}  chosen c={statistics.mean(x[0] for x in v):+.3f}  pool mean c={statistics.mean(x[1] for x in v):+.3f}  "
          f"pool SD c={statistics.mean(x[2] for x in v):.3f}  pool max c={statistics.mean(x[3] for x in v):+.3f}  frac c>0={statistics.mean(x[4] for x in v):.2f}")
tot = collections.defaultdict(float)
for f in sorted(glob.glob(f"{run}/trajectories*.jsonl")):
    for l in open(f):
        t = json.loads(l)
        if gens[t["output_id"]]["arm"] == "llm_global" and t["incremental_contrast"][t["chosen"]] is not None:
            tot[t["output_id"]] += t["incremental_contrast"][t["chosen"]]
vals = list(tot.values())
print(f"total added contrast per output (sum over units): mean {statistics.mean(vals):+.2f} nats, sd {statistics.stdev(vals):.2f}, n={len(vals)}")
