"""Key-specific attribution with the decoy-rank test, from existing detections (no model):
for every steered text, the assigned key's rank among the 100 decoy keys (correct attribution) and, for every WRONG
target key, its rank among the same decoys (false attribution).  Also the same for clean base texts (false positives).
Usage: python scripts/rank_false_attribution.py runs/<name> ..."""
import json, sys
from pathlib import Path
from collections import defaultdict

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

for run in [a for i, a in enumerate(sys.argv[1:]) if a.startswith("runs/") and (i == 0 or sys.argv[i] != "--bank")]:
    rd = Path(run)
    dets = defaultdict(dict)
    for r in load(rd / "detections_raw.jsonl"):
        if r.get("score") is not None:
            dets[r["text_hash"]][r["key_id"]] = r["score"]
    gens = [g for f in sorted(rd.glob("generations*.jsonl")) for g in load(f)]
    key_set = json.load(open(rd / "manifest.json"))["key_set"]
    targets = sorted({g["assigned_key_id"] for g in gens if g.get("assigned_key_id")})
    bank = sys.argv[sys.argv.index("--bank") + 1] if "--bank" in sys.argv else None      # e.g. decoy_rw (same-family bank)
    if bank:
        decoys = sorted(k["key_id"] for k in json.load(open(f"assets/keys_{bank}.json"))["keys"])
    else:
        decoys = sorted(k for k in next(iter(dets.values())) if k.startswith("decoy_") and not k.startswith("decoy_rw"))
    decoys = [d for d in decoys if all(d in dets[g["text_hash"]] for g in gens[:1])]
    print(f"== {rd.name}: {len(targets)} target keys, {len(decoys)} decoys")
    for arm in sorted({g["arm"] for g in gens}):
        rows = [g for g in gens if g["arm"] == arm and g.get("status", "ok") == "ok"]
        correct_top, wrong_top, wrong_n, correct_frac, wrong_frac = 0, 0, 0, [], []
        for g in rows:
            sc = dets.get(g["text_hash"], {})
            if not all(d in sc for d in decoys):
                continue
            dvals = [sc[d] for d in decoys]
            for k in targets:
                if k not in sc:
                    continue
                beaten = sum(1 for v in dvals if sc[k] > v) / len(dvals)
                if k == g["assigned_key_id"]:
                    correct_top += beaten == 1.0; correct_frac.append(beaten)
                else:
                    wrong_top += beaten == 1.0; wrong_n += 1; wrong_frac.append(beaten)
        n = len(correct_frac)
        print(f"  {arm:10s} n={n:3d}  correct key beats all decoys: {100*correct_top/max(1,n):5.1f}%  (mean frac beaten {sum(correct_frac)/max(1,n):.3f})"
              f"  | WRONG target keys beat all decoys: {100*wrong_top/max(1,wrong_n):5.2f}% of {wrong_n} pairs (mean frac {sum(wrong_frac)/max(1,wrong_n):.3f}; chance {100/(len(decoys)+1):.2f}%)")

# ---- continuous attribution statistic: per-text z of the key's score among the same-family decoys' scores -------------
# z(text, key) = (s_key - mean_d s_d) / sd_d s_d.  Null = clean (base) texts under every target key; threshold at the 1%
# conservative rank; TPR = correct key on steered texts above it; false attribution = wrong target keys above it.
import math
def zstat(sc, k, decoys):
    d = [sc[x] for x in decoys]; m = sum(d) / len(d); sd = (sum((v - m) ** 2 for v in d) / (len(d) - 1)) ** 0.5
    return (sc[k] - m) / sd if sd > 0 else 0.0
for run in [a for i, a in enumerate(sys.argv[1:]) if a.startswith("runs/") and (i == 0 or sys.argv[i] != "--bank")]:
    rd = Path(run)
    dets = defaultdict(dict)
    for r in load(rd / "detections_raw.jsonl"):
        if r.get("score") is not None:
            dets[r["text_hash"]][r["key_id"]] = r["score"]
    gens = [g for f in sorted(rd.glob("generations*.jsonl")) for g in load(f)]
    targets = sorted({g["assigned_key_id"] for g in gens if g.get("assigned_key_id")})
    bank = sys.argv[sys.argv.index("--bank") + 1] if "--bank" in sys.argv else None
    decoys = sorted(k["key_id"] for k in json.load(open(f"assets/keys_{bank}.json"))["keys"]) if bank else []
    if not decoys:
        continue
    ok = lambda sc: all(d in sc for d in decoys) and all(k in sc for k in targets)
    null = [zstat(dets[g["text_hash"]], k, decoys) for g in gens if g["arm"] == "base" and ok(dets[g["text_hash"]]) for k in targets]
    null.sort(); kq = int(math.floor(0.01 * (len(null) + 1))) - 1; thr = null[len(null) - 1 - kq] if kq >= 0 else float("inf")
    for arm in sorted({g["arm"] for g in gens} - {"base"}):
        rows = [g for g in gens if g["arm"] == arm and ok(dets[g["text_hash"]])]
        pos = [zstat(dets[g["text_hash"]], g["assigned_key_id"], decoys) for g in rows]
        wrong = [zstat(dets[g["text_hash"]], k, decoys) for g in rows for k in targets if k != g["assigned_key_id"]]
        au = sum((p > n) + 0.5 * (p == n) for p in pos for n in null) / max(1, len(pos) * len(null))
        print(f"  [decoy-z] {rd.name:28s} {arm:10s} null n={len(null)} thr@1%={thr:5.2f} | correct: mean z={sum(pos)/len(pos):5.2f} TPR@1%={sum(p > thr for p in pos)/len(pos):.2f} AUROC={au:.3f}"
              f" | wrong keys above thr: {sum(w > thr for w in wrong)/max(1,len(wrong)):.3f} (n={len(wrong)})")
