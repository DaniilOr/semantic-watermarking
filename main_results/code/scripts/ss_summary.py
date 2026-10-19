"""One table for the semantic-watermark round: cost, detection at 1% FPR (own statistic), robustness, judge, per arm."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = "runs/ss_llama3b"
PRIMARY = {"semstamp": "z_chain", "steered_sent": "z_chain", "steered_seg": "t_seg", "steered_seg_b2": "t_seg",
           "steered_seg_assist": "t_seg", "base": "z_chain", "clean_seed1": "z_chain"}
rows = defaultdict(list)
for f in glob.glob(f"{RUN}/generations.*.jsonl"):
    for l in open(f):
        g = json.loads(l); rows[g["arm"]].append(g)
det = json.load(open("runs/detect_ss_llama3b.json")) if Path("runs/detect_ss_llama3b.json").exists() else {"runs": {}}
pref = {}
for p in glob.glob(f"{RUN}/eval/api_pref_*.json"):
    pref.update(json.load(open(p)).get("per_arm", {}))
qvc = json.load(open("runs/quality_vs_clean_ss_llama3b.json")) if Path("runs/quality_vs_clean_ss_llama3b.json").exists() else {"metrics": {}}

def d(run, arm, st):
    r = det["runs"].get(run, {}).get(arm, {}).get(st)
    return f"{r['rate']:.2f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]" if r else "-"

def cost(arm):
    C = [g["cost"] for g in rows[arm] if g.get("cost")]
    if not C: return "-"
    wall = np.mean([c.get("wall_seconds", 0) for c in C])
    if "trials" in C[0]:
        tr = sum(c["trials"] for c in C); ac = sum(c["accepted"] for c in C); mx = sum(c["maxed_out"] for c in C)
        return f"{wall:5.0f}s  {tr / max(1, ac):5.1f} cand/sent  {mx / max(1, ac):4.0%} maxed"
    return f"{wall:5.0f}s"

print(f"{'arm':20s} {'n':>3s} {'med tok':>7s} {'cost':>36s} | {'stat':7s} {'clean':>18s} {'reorder':>18s} {'reseg':>18s} {'para-L':>18s} {'para-S':>18s} | {'pref':>18s} {'flaws outside band':>18s}")
for arm in ("base", "clean_seed1", "semstamp", "steered_sent", "steered_seg", "steered_seg_b2", "steered_seg_assist"):
    if arm not in rows: continue
    st = PRIMARY.get(arm, "z_chain")
    med = int(np.median([g["visible_tokens"] for g in rows[arm]]))
    p = pref.get(arm); ps = f"{p['win_rate']:.2f} [{p['win_rate_ci95'][0]:.2f},{p['win_rate_ci95'][1]:.2f}]" if p else "-"
    out = sum(1 for m in qvc["metrics"].values() if not m["arms"].get(f"ss_llama3b/{arm}", {}).get("inside_clean_band", True))
    print(f"{arm:20s} {len(rows[arm]):3d} {med:7d} {cost(arm):>36s} | {st:7s} {d('ss_llama3b', arm, st):>18s} {d('ss_llama3b_reorder_sent', arm, st):>18s} "
          f"{d('ss_llama3b_resegment', arm, st):>18s} {d('ss_llama3b_para_light', arm, st):>18s} {d('ss_llama3b_para_strong', arm, st):>18s} | {ps:>18s} {out:>18d}")
print("\nthresholds:", det.get("thresholds"), "| null texts:", det.get("n_null"))
print("all statistics per arm (clean):")
for arm, r in det["runs"].get("ss_llama3b", {}).items():
    print(f"  {arm:20s} " + "  ".join(f"{st}: {r[st]['rate']:.2f} auroc {r[st]['auroc']:.3f} wrong {r[st]['wrong_key_rate']:.3f}" for st in ("z_chain", "t_cont", "t_seg") if st in r))
