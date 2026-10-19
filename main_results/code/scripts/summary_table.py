"""One consolidated table per run (poe_steer arm) from results.json, t_null_pool.json and the same-family decoy bank:
detect = t-detector TPR at 1% (pooled clean null); FA = t-detector flag rate under a wrong key; attr = correct key beats
all same-family decoys; attr_z = decoy-standardized z attribution at 1%; cost = paired mean log-lik/token delta.
Usage: python scripts/summary_table.py --bank decoy_rw runs/a runs/b ...  (--md for markdown)"""
import json, math, sys
from pathlib import Path
from collections import defaultdict
md = "--md" in sys.argv
bank = sys.argv[sys.argv.index("--bank") + 1] if "--bank" in sys.argv else "decoy_rw"
runs = [a for i, a in enumerate(sys.argv[1:]) if a.startswith("runs/") and sys.argv[i] != "--bank"]
decoys = sorted(k["key_id"] for k in json.load(open(f"assets/keys_{bank}.json"))["keys"])
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def zstat(sc, k):
    d = [sc[x] for x in decoys]; m = sum(d) / len(d); sd = (sum((v - m) ** 2 for v in d) / (len(d) - 1)) ** 0.5
    return (sc[k] - m) / sd if sd > 0 else 0.0
rows = []
for run in runs:
    rd = Path(run); name = rd.name
    try: res = json.load(open(rd / "eval" / "results.json"))
    except FileNotFoundError: continue
    a = res["per_arm"].get("poe_steer"); 
    if not a: continue
    q = a.get("quality_proxy", {})
    tp = {}
    p = rd / "eval" / "t_null_pool.json"
    if p.exists():
        tp = json.load(open(p))["per_arm"].get("poe_steer", {})
    dets = defaultdict(dict)
    for r in load(rd / "detections_raw.jsonl"):
        if r.get("score") is not None: dets[r["text_hash"]][r["key_id"]] = r["score"]
    gens = load(rd / "generations.jsonl"); targets = sorted({g["assigned_key_id"] for g in gens if g.get("assigned_key_id")})
    ok = lambda sc: all(d in sc for d in decoys) and all(k in sc for k in targets)
    attr = attr_z = fa_rank = None
    if any(ok(dets[g["text_hash"]]) for g in gens):
        pos_rows = [g for g in gens if g["arm"] == "poe_steer" and ok(dets[g["text_hash"]])]
        attr = sum(all(dets[g["text_hash"]][g["assigned_key_id"]] > dets[g["text_hash"]][d] for d in decoys) for g in pos_rows) / len(pos_rows)
        null = sorted(zstat(dets[g["text_hash"]], k) for g in gens if g["arm"] == "base" and ok(dets[g["text_hash"]]) for k in targets)
        kq = int(math.floor(0.01 * (len(null) + 1))) - 1; thr = null[len(null) - 1 - kq] if kq >= 0 else float("inf")
        attr_z = sum(zstat(dets[g["text_hash"]], g["assigned_key_id"]) > thr for g in pos_rows) / len(pos_rows)
        wrong = [zstat(dets[g["text_hash"]], k) for g in pos_rows for k in targets if k != g["assigned_key_id"]]
        fa_rank = sum(w > thr for w in wrong) / len(wrong)
    rows.append({"run": name, "detect": tp.get("tpr_base_alpha0.01"), "FA_t": tp.get("false_attribution_rate_alpha0.01"),
                 "auroc_t": (a.get("t_detector") or {}).get("auroc") if False else ((res.get("t_detector") or {}).get("per_arm", {}).get("poe_steer", {}).get("auroc_t_vs_base") or {}).get("auroc"),
                 "attr": attr, "attr_z": attr_z, "FA_z": fa_rank, "cost": (q.get("paired_delta_vs_base") or {}).get("mean"),
                 "leak": (a.get("leakage") or {}).get("any_term_frac"), "loops": q.get("loop8_frac"), "len": a.get("visible_tokens_mean")})
cols = ["run", "detect", "FA_t", "auroc_t", "attr", "attr_z", "FA_z", "cost", "leak", "loops", "len"]
def f(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)
if md:
    print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in rows: print("| " + " | ".join(f(r[c]) for c in cols) + " |")
else:
    w = {c: max(len(c), *(len(f(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows: print("  ".join(f(r[c]).ljust(w[c]) for c in cols))
