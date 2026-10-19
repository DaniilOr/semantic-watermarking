"""GSM8K accuracy of every arm in runs/it_fix_llama3b (shard files AND the copied plain / SemStamp / push files), on the
prompts all arms share.  Read-only; prints a table."""

import glob
import json
import sys
from collections import defaultdict

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from it_eval import gsm8k_correct  # noqa: E402
from wmworld.posstats import wilson  # noqa: E402

prompts = {json.loads(l)["prompt_id"]: json.loads(l) for l in open("data/it_bench/prompts.jsonl", encoding="utf-8") if '"test"' in l}
recs = defaultdict(dict)
for f in glob.glob("runs/it_fix_llama3b/generations.*.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip().endswith("}"):
            r = json.loads(l)
            if r.get("status", "ok") == "ok" and r.get("domain") == "it_gsm8k":
                recs[r["arm"]][r["prompt_id"]] = r
for label, keep in (("all prompts of each arm", None), ("prompts shared by all arms", set.intersection(*(set(v) for v in recs.values())))):
    print(f"-- {label}" + (f" (n={len(keep)})" if keep else ""))
    for arm in sorted(recs):
        rs = [r for p, r in recs[arm].items() if keep is None or p in keep]
        k = sum(gsm8k_correct(r.get("text") or "", prompts[r["prompt_id"]]["gold"])[1] for r in rs); n = len(rs)
        lo, hi = wilson(k, n)
        print(f"{arm:22s} {100 * k / n:5.1f}% [{100 * lo:.1f},{100 * hi:.1f}] n {n}")
