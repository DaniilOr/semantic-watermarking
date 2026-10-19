"""Count texts in which generation continued past the model's end of turn (the decoded text contains a fresh chat-header
word 'assistant' / 'user' on its own, e.g. '```assistant\\n\\n'), per run x arm x domain."""
import glob, json, re, sys
from collections import defaultdict
PAT = re.compile(r"(^|[^\w])(assistant|user)\n\n|<\|im_start\|>|<\|eot_id\|>|<\|start_header_id\|>")
runs = sys.argv[1:] or sorted(p for p in glob.glob("runs/*_bench_*") if not re.search(r"_(para_light|para_strong|reorder_sent|resegment|null)$", p))
for run in runs:
    c = defaultdict(lambda: [0, 0, 0])                                     # n, leak, hit max_new (>=195 visible tokens)
    for f in glob.glob(f"{run}/generations.*.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            if r.get("status", "ok") != "ok" or not r.get("text"):
                continue
            k = (r["arm"], r.get("domain"))
            c[k][0] += 1; c[k][1] += bool(PAT.search(r["text"])); c[k][2] += (r.get("visible_tokens") or 0) >= 195
    for (arm, dom), (n, l, m) in sorted(c.items()):
        print(f"{run:28s} {arm:18s} {dom:14s} n {n:5d}  leak {l:5d} ({100*l/n:5.1f}%)  at-budget {100*m/n:5.1f}%")
