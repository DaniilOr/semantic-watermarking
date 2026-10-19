"""Per-domain detection and attribution for a run: within-text statistic at the pooled clean 1% threshold, correct key
beating all same-construction decoys, mean length, key-word mentions.  Usage: python scripts/per_domain.py runs/<name> --bank decoy_rw"""
import json, sys
from collections import defaultdict
from pathlib import Path
rd = Path(sys.argv[1]); bank = sys.argv[sys.argv.index("--bank") + 1] if "--bank" in sys.argv else "decoy_rw"
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
tp = json.load(open(rd / "eval" / "t_null_pool.json"))["per_arm"]
thr = tp[[a for a in tp if a != "base"][0]]["thr_base_alpha0.01"]
decoys = sorted(k["key_id"] for k in json.load(open(f"assets/keys_{bank}.json"))["keys"])
t, sc = {}, defaultdict(dict)
for r in load(rd / "detections_raw.jsonl"):
    if r.get("t_stat") is not None: t[(r["text_hash"], r["key_id"])] = r["t_stat"]
    if r.get("score") is not None: sc[r["text_hash"]][r["key_id"]] = r["score"]
gens = [g for f in sorted(rd.glob("generations*.jsonl")) for g in load(f)]
seen = set(); rows = defaultdict(list)
for g in gens:
    if g["id"] in seen: continue
    seen.add(g["id"]); rows[(g["domain"], g["arm"])].append(g)
print(f"pooled clean threshold: T > {thr:.2f}")
print(f"{'domain':18s} {'arm':11s} {'n':>4s} {'tokens':>6s} {'detected@1%':>11s} {'beats all 100':>13s} {'median T':>8s}")
for (dom, arm), gs in sorted(rows.items()):
    tv = [t[(g["text_hash"], g["assigned_key_id"])] for g in gs if (g["text_hash"], g["assigned_key_id"]) in t]
    ba = [all(sc[g["text_hash"]][g["assigned_key_id"]] > sc[g["text_hash"]][d] for d in decoys) for g in gs
          if g["assigned_key_id"] in sc[g["text_hash"]] and all(d in sc[g["text_hash"]] for d in decoys)]
    tok = sum(g["visible_tokens"] for g in gs) / len(gs)
    med = sorted(tv)[len(tv) // 2] if tv else float("nan")
    print(f"{dom:18s} {arm:11s} {len(gs):4d} {tok:6.0f} {100 * sum(x > thr for x in tv) / max(1, len(tv)):10.1f}% {100 * sum(ba) / max(1, len(ba)):12.1f}% {med:8.2f}")
