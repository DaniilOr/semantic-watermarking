"""Per monitor of GRPO runs: how many of the 12 stored sample sentences are degenerate (<= 3 characters, e.g. ':')."""
import json, sys
for run in sys.argv[1:]:
    d = json.load(open("runs/ss_rl/%s/rl_log.json" % run))
    row = []
    for mo in d["monitor"]:
        if not mo.get("adapter"):
            continue
        short = sum(len(s["sentence"].strip()) <= 3 for s in mo["samples"])
        row.append("%s:%d/%d" % (mo["step"], short, len(mo["samples"])))
    print(run, " ".join(row))
