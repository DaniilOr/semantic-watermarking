import json, random, sys
rows = [json.loads(l) for l in open("runs/code/eval/code_eval.jsonl") if l.strip()]
by = {}
for r in rows: by.setdefault(r["arm"], {})[r["prompt_id"]] = r["status"] == "pass"
base = by["base"]; random.seed(0)
for arm in ("wm_neutral", "wm_decoy"):
    pids = [p for p in by[arm] if p in base]
    d = [by[arm][p] - base[p] for p in pids]
    boots = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
    print(f"{arm}: paired pass-rate delta {sum(d)/len(d):+.3f}  95% CI [{boots[100]:+.3f}, {boots[3899]:+.3f}]  n={len(d)}")
