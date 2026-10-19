"""Long stories: preference (win rate vs clean) and pointwise quality per arm, paired with the clean text of the same prompt."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/long_story")
SLUG = "google_gemma-4-31b-it"
METRICS = {
    "quality 1-5 (higher better)": lambda j: float(j["quality"]),
    "major flaw share": lambda j: 1.0 if j["severity"] == "major" else 0.0,
    "flaws per text": lambda j: float(len(set(j["flaws"]))),
    "not fully on task": lambda j: 0.0 if j["addresses_task"] == "yes" else 1.0,
    "insertions or drift": lambda j: 1.0 if {"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"]) else 0.0,
    "repetition or loops": lambda j: 1.0 if "repetition_or_loops" in j["flaws"] else 0.0,
}
rng = np.random.default_rng(0)
pref = json.load(open(run / "eval" / f"api_pref_{SLUG}.json"))
print(f"== preference vs the clean text of the same prompt (judge sees {pref.get('max_words')} words max, both orders) ==")
for arm, s in pref["per_arm"].items():
    print(f"  {arm:8s} n={s['n_prompts']} win rate {s['win_rate']:.3f} [{s['win_rate_ci95'][0]:.3f}, {s['win_rate_ci95'][1]:.3f}] ties {s['tie_share']:.2f}")
rows = [json.loads(l) for l in open(run / "eval" / f"api_pointwise_{SLUG}.jsonl")]
by = defaultdict(dict)
for r in rows:
    by[r["arm"]][r["prompt_id"]] = r["judgment"]
gens = [json.loads(l) for f in run.glob("generations*.jsonl") for l in open(f)]
words = defaultdict(list)
for g in gens:
    if g.get("status", "ok") == "ok":
        words[g["arm"]].append(len(g["text"].split()))
print("\n== length (words): " + ", ".join(f"{a} median {int(np.median(w))}" for a, w in sorted(words.items())))
out = {"preference": pref["per_arm"], "quality": {}}
print("\n== pointwise quality (one text per call, full length), paired difference vs clean on the same prompts ==")
for name, fn in METRICS.items():
    print(f"  {name}")
    out["quality"][name] = {}
    for arm in sorted(by):
        pids = sorted(set(by[arm]) & set(by["base"]))
        v = np.array([fn(by[arm][p]) for p in pids])
        line = f"    {arm:8s} n={len(pids):3d} mean {v.mean():.3f}"
        rec = {"n": len(pids), "mean": float(v.mean())}
        if arm != "base":
            d = v - np.array([fn(by["base"][p]) for p in pids])
            boot = [float(d[rng.integers(0, len(d), size=len(d))].mean()) for _ in range(4000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rec.update({"diff_vs_clean": float(d.mean()), "diff_ci95": [float(lo), float(hi)]})
            line += f"   diff vs clean {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("  *" if lo > 0 or hi < 0 else "")
        out["quality"][name][arm] = rec
        print(line)
json.dump(out, open("runs/long_story_quality.json", "w"), indent=1)
print("\nwrote runs/long_story_quality.json")
