"""Old vs new adapter decoding on the same prompts: wall seconds per text, tokens, candidates per unit, kept in mask."""
import glob, json, sys
import numpy as np


def load(run, arm):
    out = {}
    for f in glob.glob(f"{run}/generations.{arm}.shard*.jsonl"):
        for ln in open(f):
            g = json.loads(ln); out[g["prompt_id"]] = g
    return out


new, old, ref = load("runs/ob_speedtest", "chain_grpo300"), load("runs/ob_bench", "chain_grpo300"), load("runs/ob_bench", "chain_reference")
pids = sorted(p for p in new if p in old)
for d, name in ((old, "grpo300 old code"), (new, "grpo300 new code"), (ref, "sentence+push+margin")):
    gs = [d[p] for p in pids if p in d]
    w = [g["cost"]["wall_seconds"] for g in gs]
    U = [u for g in gs for u in g["cost"]["units"] if u.get("prev_bin") is not None and "n_drawn" in u]
    tok = np.mean([g["visible_tokens"] for g in gs]); cpu = np.mean([u["n_drawn"] for u in U]); km = np.mean([u["in_mask"] for u in U])
    print("%-22s texts %3d  s/text mean %6.1f median %6.1f  tokens %5.1f  cand/unit %5.2f  kept in mask %.3f" % (name, len(gs), np.mean(w), np.median(w), tok, cpu, km))
