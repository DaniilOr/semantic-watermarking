import glob, json, sys
import numpy as np
for run in sys.argv[1:]:
    gens = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(f"{run}/generations*.jsonl") for l in open(f)}
    trs = [json.loads(l) for f in glob.glob(f"{run}/trajectories*.jsonl") for l in open(f)]
    p = [t for t in trs if t["scale_status"] == "poe"]
    ended = [t for t in p if t["chosen_ended"]]
    eos_c = [t["cost"]["per_token"]["contrast"][-1] for t in ended]
    eos_b = [t["cost"]["per_token"]["beta_t"][-1] for t in ended]
    other = [x for t in p for x in t["cost"]["per_token"]["contrast"][:-1]]
    print(f"{run}: outputs={len(p)} ended_by_eos={len(ended)} lengths={[gens[t['output_id']]['visible_tokens'] for t in ended]} "
          f"EOS-step contrast={np.round(eos_c,3).tolist()} beta_t={np.round(eos_b,2).tolist()} | other tokens contrast mean {np.mean(other):+.3f}")
