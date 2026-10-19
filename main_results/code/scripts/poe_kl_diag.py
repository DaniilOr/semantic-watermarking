import glob, json, sys
import numpy as np
for run in sys.argv[1:]:
    trs = [json.loads(l) for f in glob.glob(f"{run}/trajectories*.jsonl") for l in open(f)]
    p = [t for t in trs if t["scale_status"] == "poe"]
    kl = np.array([x for t in p for x in t["cost"]["per_token"].get("kl", [])])
    b = np.array([x for t in p for x in t["cost"]["per_token"]["beta_t"]])
    d = np.array([x for t in p for x in t["cost"]["per_token"]["contrast"]])
    lp = np.array([x for t in p for x in t["cost"]["per_token"]["logp_g"]])
    print(f"{run}: outputs={len(p)} tokens={kl.size} KL: mean {kl.mean():.4f} max {kl.max():.4f} q95 {np.quantile(kl,0.95):.4f} | "
          f"beta_t: mean {b.mean():.2f} max {b.max():.1f} frac==beta_max {np.mean(b>=b.max()-1e-9):.2f} | chosen contrast mean {d.mean():+.4f} | "
          f"chosen logp_g mean {lp.mean():.3f}")
