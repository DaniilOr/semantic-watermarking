"""Where shipped SemStamp's candidates go (runs/ss_llama3b, arm semstamp): bin rejects vs margin rejects vs forced."""
import json, glob
import numpy as np
T = dict(trials=0, accepted=0, maxed_out=0, margin_rejects=0, bin_rejects=0)
inm = []
for f in glob.glob("runs/ss_llama3b/generations.semstamp.shard*.jsonl"):
    for ln in open(f):
        c = json.loads(ln)["cost"]
        for k in T:
            T[k] += c.get(k, 0)
        inm += [s["in_mask"] for s in c.get("sentence_trace", [])]
cand_in = T["margin_rejects"] + (T["accepted"] - T["maxed_out"])          # candidates that landed in the mask
print(f"candidates drawn {T['trials']}, sentences accepted {T['accepted']} -> {T['trials'] / T['accepted']:.1f} candidates per sentence")
print(f"per candidate: in the mask {cand_in / T['trials']:.3f}; of those in the mask, rejected by the margin {T['margin_rejects'] / cand_in:.3f}")
print(f"rejected for the bin {T['bin_rejects']} ({T['bin_rejects'] / T['trials']:.3f} of candidates), for the margin {T['margin_rejects']} ({T['margin_rejects'] / T['trials']:.3f})")
print(f"sentences forced through after 100 tries: {T['maxed_out'] / T['accepted']:.3f}; accepted sentences in the mask: {np.mean(inm):.3f}")

# the bounded chain arms, same accounting (every drawn candidate is logged with its projections)
import sys
sys.path.insert(0, ".")
from wmworld import semwm
key = semwm.Key.shipped(); planes = semwm.Hyperplanes(np.zeros((3, 1)) + 1)   # only bin_of on logged projections is used
for arm in ("chain_sent_blk4x16", "chain_seg_blk4x16", "chain_sent_blk4x16_push1", "chain_seg_blk4x16_push1", "chain_sent_blk4x16_m0.02", "chain_seg_blk4x16_m0.02", "chain_sent_blk4x16_push1_m0.02", "chain_seg_blk4x16_push1_m0.02"):
    n_c = n_in = n_u = n_force = n_margin_ok = 0
    for f in glob.glob(f"runs/ss_llama3b/generations.{arm}.shard*.jsonl"):
        for ln in open(f):
            for u in json.loads(ln)["cost"]["units"]:
                if u["prev_bin"] is None:
                    continue
                mask = semwm.valid_mask(key, u["prev_bin"])
                C = np.asarray(u["cands"])[:, :3]
                ins = [planes.bin_of(p) in mask for p in C]
                n_c += len(C); n_in += sum(ins); n_u += 1; n_force += not any(ins)
                n_margin_ok += sum(1 for p, i in zip(C, ins) if i and np.min(np.abs(p)) >= semwm.MARGIN)
    print(f"{arm}: {n_c / n_u:.2f} candidates per unit; per candidate in the mask {n_in / n_c:.3f}; "
          f"in the mask AND clearing SemStamp's margin {n_margin_ok / n_c:.4f}; units with no in-mask candidate after 16: {n_force / n_u:.3f}")
