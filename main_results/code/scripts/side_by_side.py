"""Side-by-side outputs: for N prompts of a run, print the task and each arm's output (plus MBPP pass/fail when available)."""
import json, sys, textwrap
from pathlib import Path
run = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 3; maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
gens = []
for f in sorted(run.glob("generations*.jsonl")):
    gens += [json.loads(l) for l in open(f) if l.strip()]
status = {}
p = run / "eval" / "code_eval.jsonl"
if p.exists():
    for l in open(p):
        r = json.loads(l); status[r["id"]] = r["status"]
prompts = {}
for f in Path("data/prepared").glob("*.jsonl"):
    for l in open(f):
        d = json.loads(l); prompts[d["prompt_id"]] = d
by = {}
for g in gens: by.setdefault(g["prompt_id"], {})[g["arm"]] = g
shown = 0
for pid in sorted(by, key=lambda x: hash(x)):
    arms = by[pid]
    if "base" not in arms or len(arms) < 3: continue
    if any(len(a["text"]) > maxc for a in arms.values()): continue
    print("=" * 100); print(f"PROMPT {pid}  (key {arms['base'].get('assigned_key_id')})"); print(textwrap.indent(prompts[pid]["prompt"][:700], "  "))
    for arm in ("base", "wm_neutral", "wm_decoy"):
        g = arms[arm]; st = status.get(g["id"], "")
        print(f"--- {arm} ({g['visible_tokens']} tok{', ' + st if st else ''}):"); print(textwrap.indent(g["text"], "    "))
    shown += 1
    if shown >= n: break
