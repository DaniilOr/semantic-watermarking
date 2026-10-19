"""Side-by-side original (base) vs watermarked (llm_global) outputs per prompt, from a run's generations (no scores needed)."""
import csv, glob, json, sys
from pathlib import Path
run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/dev")
arm_w = sys.argv[2] if len(sys.argv) > 2 else "llm_global"
gens = [json.loads(l) for f in sorted(run.glob("generations*.jsonl")) for l in open(f)]
keys = {k["key_id"]: k for k in json.load(open("assets/keys_development.json"))["keys"]}
prompts = {}
for f in Path("data/prepared").glob("*.jsonl"):
    for l in open(f):
        d = json.loads(l); prompts[d["prompt_id"]] = d
by = {}
for g in gens:
    by.setdefault(g["prompt_id"], {})[g["arm"]] = g
md = [f"# 1-1 comparison: original (`base`) vs watermarked (`{arm_w}`) — run `{run.name}`\n",
      "Same prompt, same seed, same assigned key. `base` never sees the key; the watermarked text is steered by "
      "candidate selection against the key-conditioned world. Scores are added by `evaluate` (see eval/samples.md).\n"]
rows = []
for pid in sorted(by, key=lambda p: (prompts.get(p, {}).get("domain", ""), p)):
    a = by[pid]
    if "base" not in a or arm_w not in a:
        continue
    b, w = a["base"], a[arm_w]
    k = keys.get(w.get("key_id") or w.get("assigned_key_id"), {})
    p = prompts.get(pid, {})
    md.append(f"\n---\n\n## {pid} ({p.get('domain')})\n\n**Task:**\n\n> " + p.get("prompt", "").replace("\n", "\n> ") +
              f"\n\n**Key `{k.get('key_id')}`** ({k.get('family')}): `{k.get('prefix')}`\n")
    md.append(f"\n### original / base ({b['visible_tokens']} tokens, stop={b['stop_reason']})\n\n```text\n{b['text'].replace('```', chr(39)*3)}\n```\n")
    md.append(f"\n### watermarked / {arm_w} ({w['visible_tokens']} tokens, stop={w['stop_reason']})\n\n```text\n{w['text'].replace('```', chr(39)*3)}\n```\n")
    rows.append([pid, p.get("domain"), k.get("key_id"), k.get("family"), k.get("prefix"), p.get("prompt"), b["visible_tokens"], b["text"], w["visible_tokens"], w["text"]])
out_md = run / f"pairs_base_vs_{arm_w}.md"; out_md.write_text("\n".join(md), encoding="utf-8")
with (run / f"pairs_base_vs_{arm_w}.csv").open("w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["prompt_id", "domain", "key_id", "family", "key_prefix", "prompt", "base_tokens", "base_text", f"{arm_w}_tokens", f"{arm_w}_text"]); wr.writerows(rows)
print(f"wrote {out_md} and .csv with {len(rows)} pairs")
