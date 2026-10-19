"""Per-key detection by seed-phrase length stratum (short = 4 words, long = 10 words) from t_null_pool-style data:
mean t under the assigned key and beats-all-decoys rate per key, for the poe_steer arm of the given runs."""
import json, sys
from pathlib import Path
from collections import defaultdict
keys = {k["key_id"]: k for k in json.load(open("assets/keys_randomwords.json"))["keys"]}
for run in sys.argv[1:]:
    rd = Path(run)
    res = json.load(open(rd / "eval" / "results.json"))
    dets = {}
    for l in open(rd / "detections_raw.jsonl"):
        r = json.loads(l); dets[(r["text_hash"], r["key_id"])] = r
    gens = [json.loads(l) for l in open(rd / "generations.jsonl") if l.strip()]
    by = defaultdict(lambda: {"t": [], "n": 0})
    for g in gens:
        if g["arm"] != "poe_steer": continue
        d = dets.get((g["text_hash"], g["assigned_key_id"]))
        if d and d.get("t_stat") is not None:
            by[g["assigned_key_id"]]["t"].append(d["t_stat"])
    strata = defaultdict(list)
    print(f"== {rd.name}")
    for kid, v in sorted(by.items()):
        k = keys[kid]; mt = sum(v["t"]) / len(v["t"])
        strata[k["length_stratum"]].append(mt)
        print(f"  {kid} {k['length_stratum']:5s} tok={k['token_length']:2d} n={len(v['t'])} mean_t={mt:5.2f}  {k['prefix']}")
    for s, v in strata.items():
        print(f"  stratum {s}: mean t {sum(v)/len(v):.2f} over {len(v)} keys")
