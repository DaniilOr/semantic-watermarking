"""Per dataset (Llama-3.2-3B chat): candidates per kept sentence and mean wall seconds per text for SemStamp (trials /
accepted), Push + Margin (draws per sentence after the first, as push_cost_cols.py) and PMark (draws / sentences), plus
the judge win rate vs plain for SemStamp and push (per_arm_domain of the preference judge).  Read-only; prints JSON."""
import glob, json, statistics as St
SETS = [("sp_bench_llama3b", "pm_sp_bench_llama3b", ["sp_realnews", "sp_booksum"]),
        ("ob_bench", "pm_ob_bench", ["ob_arxiv", "ob_booksum", "ob_wikipedia"]),
        ("it_bench_llama3b", "pm_it_bench_llama3b", ["it_writingprompts", "it_dolly", "it_gsm8k", "it_ifeval"])]
out = {}
def recs(run, arm, dom):
    for f in glob.glob(f"runs/{run}/generations.{arm}.shard*.jsonl"):
        for l in open(f, encoding="utf-8"):
            if not l.strip().endswith("}"):
                continue
            r = json.loads(l)
            if r.get("domain") == dom and r.get("status", "ok") == "ok":
                yield r.get("cost") or {}
for run, pmrun, doms in SETS:
    judge = json.load(open(f"runs/{run}/eval/api_pref_google_gemma-4-31b-it.json")).get("per_arm_domain", {})
    for d in doms:
        row = {}
        tr = ac = 0; ws = []
        for c in recs(run, "semstamp", d):
            tr += c.get("trials", 0); ac += c.get("accepted", 0); ws.append(c.get("wall_seconds", 0))
        row["semstamp"] = {"cand": tr / ac if ac else None, "sec": St.mean(ws) if ws else None}
        dr = un = 0; ws = []
        for c in recs(run, "chain_reference", d):
            ws.append(c.get("wall_seconds", 0))
            for u in (c.get("units") or [])[1:]:
                un += 1; dr += u["n_drawn"]
        row["push"] = {"cand": dr / un if un else None, "sec": St.mean(ws) if ws else None}
        ca = ac = 0; ws = []
        for c in recs(pmrun, "pmark", d):
            ca += c.get("candidates", 0); ac += c.get("accepted", 0); ws.append(c.get("wall_seconds", 0))
        row["pmark"] = {"cand": ca / ac if ac else None, "sec": St.mean(ws) if ws else None}
        dk = d.replace("it_", "it_") 
        for arm, k in (("semstamp", "semstamp"), ("chain_reference", "push")):
            j = judge.get(f"{arm}|{d}")
            row[k]["judge"] = j["win_rate"] if j else None
            row[k]["judge_ci"] = j["win_rate_ci95"] if j else None
        out[d] = row
print(json.dumps(out, indent=1))
