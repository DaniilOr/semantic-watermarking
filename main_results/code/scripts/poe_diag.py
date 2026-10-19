"""Per-token diagnostics of PoE runs: distribution of the chosen-token key/neutral contrast and of the entropy gate."""
import glob, json, sys, statistics
import numpy as np
for run in sys.argv[1:]:
    trs = [json.loads(l) for f in glob.glob(f"{run}/trajectories*.jsonl") for l in open(f)]
    p = [t for t in trs if t["scale_status"] == "poe"]
    d = np.array([x for t in p for x in t["cost"]["per_token"]["contrast"]])
    h = np.array([x for t in p for x in t["cost"]["per_token"]["entropy"]])
    b = np.array([x for t in p for x in t["cost"]["per_token"]["beta_t"]])
    print(f"{run}: outputs={len(p)} tokens={d.size}  contrast(chosen): mean {d.mean():+.4f} sd {d.std():.4f} "
          f"q05 {np.quantile(d,0.05):+.3f} q50 {np.quantile(d,0.5):+.3f} q95 {np.quantile(d,0.95):+.3f} frac>0 {np.mean(d>0):.2f} | "
          f"entropy: median {np.median(h):.2f} q25 {np.quantile(h,0.25):.2f} q75 {np.quantile(h,0.75):.2f} frac<0.5 {np.mean(h<0.5):.2f} frac>2 {np.mean(h>2):.2f} | "
          f"beta_t mean {b.mean():.2f} frac steered {np.mean(b>0):.2f}")
