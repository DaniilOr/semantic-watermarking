"""Where do the 1B-draft chain arms lose in-mask units: drafting, the 3B sampler check, or repairs?"""
import json, glob, collections
import numpy as np
import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/daniil.orel/wmworld/runs/ss_chain_smoke"
arms = sorted({f.split("generations.")[1].split(".shard")[0] for f in glob.glob(f"{RUN}/generations.chain_*_d1b*.jsonl")})
for arm in arms:
    U = []
    for f in glob.glob(f"{RUN}/generations.{arm}.shard*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            U += [u for u in g["cost"]["units"] if u["prev_bin"] is not None]
    n = len(U)
    r0 = [u["rounds"][0] for u in U if u.get("rounds")]
    draft_in = np.mean([r["in_mask"] for r in r0])                       # share of 1B drafts in the mask
    any_in = np.mean([r["depth_top"] > 0 for r in r0])                   # at least one of 8 drafts in the mask
    top_ok = np.mean([r["viol"][0] == -1 for r in r0])                   # deepest draft passes the 3B check
    top_in_ok = np.mean([r["viol"][0] == -1 for r in r0 if r["depth_top"] > 0])
    # accepted in round 0 but NOT the deepest draft (deepest failed the check, a shallower one passed)
    took_other = [u for u in U if u.get("rounds") and u.get("round") == 0 and u["rounds"][0]["viol"][0] != -1 and not u.get("eos_repair")]
    other_in = np.mean([u["in_mask"] for u in took_other]) if took_other else float("nan")
    rep = [u for u in U if u.get("round", 0) >= 1]
    rep_in = np.mean([u["in_mask"] for u in rep]) if rep else float("nan")
    r0_top = [u for u in U if u.get("round") == 0 and u["rounds"][0]["viol"][0] == -1]
    top_in = np.mean([u["in_mask"] for u in r0_top]) if r0_top else float("nan")
    vpos = [r["viol"][0] / r["len"][0] for r in r0 if r["viol"][0] != -1]
    lens = [r["len"][0] for r in r0]
    print(f"\n{arm}: {n} units | final in-mask {np.mean([u['in_mask'] for u in U]):.2f}")
    print(f"  1B drafts in mask (per draft) {draft_in:.2f}; >=1 of 8 in mask {any_in:.2f}")
    print(f"  deepest draft passes 3B check: {top_ok:.2f} (when it is in the mask: {top_in_ok:.2f}); mean unit len {np.mean(lens):.1f} tok; first bad token at {np.mean(vpos):.2f} of the unit")
    print(f"  outcome: deepest kept {len(r0_top)/n:.2f} (in mask {top_in:.2f}) | shallower draft kept instead {len(took_other)/n:.2f} (in mask {other_in:.2f}) | repaired {len(rep)/n:.2f} (in mask {rep_in:.2f})")
    paths = collections.Counter(u.get("path", "v1") for u in U)
    if "v1" not in paths:
        print("  v2 paths: " + ", ".join(f"{k} {v / n:.2f} (in mask {np.mean([u['in_mask'] for u in U if u.get('path') == k]):.2f})" for k, v in paths.most_common())
              + f"; substitutions/unit {np.mean([u.get('subs', 0) for u in U]):.2f}")
