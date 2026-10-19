"""Same-prompt comparison of an adapter arm that is still generating (default chain_grpo300) with the finished arms.

Prompts: those of the given dataset for which the adapter arm has a non-empty text.  On exactly these prompts, per arm:
detection (sentence chain z against the frozen per-dataset thresholds of runs/ob_detect.json: human-null 1% / 0.1%),
detector-side in-mask rate, generation cost from the records (seconds, candidates per sentence, first-block per-draw
hit rates, kept sentence in mask / clear of 0.02), judge preference vs base (both orders) and pointwise flaws.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.posstats import wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="chain_grpo300")
ap.add_argument("--domain", default="ob_arxiv")
ap.add_argument("--compare", nargs="*", default=["base", "clean_seed1", "semstamp", "chain_reference", "chain_seg_push_m"])
ap.add_argument("--out", default="runs/grpo_partial_stats.json")
a = ap.parse_args()
EMB = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"

recs = defaultdict(dict)                                        # arm -> prompt_id -> record
for f in glob.glob("runs/ob_bench/generations.*.shard*.jsonl"):
    for ln in open(f):
        g = json.loads(ln)
        if g.get("domain") == a.domain:
            recs[g["arm"]][g["prompt_id"]] = g
pids = sorted(p for p, g in recs[a.arm].items() if g.get("text"))
arms = [a.arm] + [x for x in a.compare if x in recs]
th = json.load(open("runs/ob_detect.json"))["thresholds"]
key = semwm.Key.shipped(); emb = semwm.Embedder(EMB, device="cpu", batch_size=128)
planes = semwm.Hyperplanes.for_key(key, emb.dim); signs = semwm.Hyperplanes(np.eye(3))
pref = defaultdict(lambda: defaultdict(list))                   # arm -> prompt -> [score per order]
for ln in open("runs/ob_bench/eval/api_pref_google_gemma-4-31b-it.jsonl"):
    r = json.loads(ln); b = r["judgment"]["better"]
    pref[r["arm"]][r["prompt_id"]].append(0.5 if b == "tie" else float(b == r["wm_is"]))
pw = defaultdict(dict)
for ln in open("runs/ob_bench/eval/api_pointwise_google_gemma-4-31b-it.jsonl"):
    r = json.loads(ln); pw[r["arm"]][r["prompt_id"]] = r["judgment"]
out = {"arm": a.arm, "domain": a.domain, "n_prompts": len(pids), "arms": {}}
print(f"{a.arm}: {len(pids)} {a.domain} prompts with a non-empty text; every arm below is read on exactly these prompts\n")
for arm in arms:
    gs = [recs[arm][p] for p in pids if p in recs[arm]]
    ok = [g for g in gs if g.get("text")]
    sc = [semwm.score_text(g["text"], key, emb, planes=planes) for g in ok]
    z = [s["z_chain"] for s in sc]
    zz = [x if x is not None else -np.inf for x in z]
    rec = {"n": len(gs), "empty": len(gs) - len(ok)}
    for t in ("human_1%", "human_0.1%"):
        thr = th[f"{a.domain}:z_chain"][t]; k = sum(x > thr for x in zz)
        rec[f"tpr_{t}"] = k / len(gs); rec[f"tpr_{t}_ci"] = wilson(k, len(gs))  # empty texts count as misses
    rec["mean_z"] = float(np.mean([x for x in z if x is not None]))
    rec["detector_in_mask"] = float(np.mean([m for s in sc for m in s["in_mask"]]))
    c = [g["cost"] for g in ok]
    rec["s_per_text"] = float(np.mean([x.get("wall_seconds") or 0 for x in c])); rec["tokens"] = float(np.mean([g["visible_tokens"] for g in ok]))
    U = [u for x in c for u in x.get("units", []) if u.get("prev_bin") is not None and "n_drawn" in u]
    if U:
        fb = [(semwm.signed_distance(np.array(cd[:3]), semwm.valid_mask(key, u["prev_bin"]), signs), np.min(np.abs(cd[:3])))
              for u in U for cd in u["cands"][:4]]
        kept = [u["cands"][u["chosen"]][:3] for u in U]
        rec.update({"cand_per_unit": float(np.mean([u["n_drawn"] for u in U])),
                    "first_block_in_mask": float(np.mean([d > 0 for d, _ in fb])),
                    "first_block_clear": float(np.mean([d > 0 and m >= 0.02 for d, m in fb])),
                    "kept_in_mask": float(np.mean([u["in_mask"] for u in U])),
                    "kept_clear": float(np.mean([u["in_mask"] and min(abs(v) for v in q) >= 0.02 for u, q in zip(U, kept)]))})
    if "trials" in c[0]:
        rec["cand_per_unit"] = sum(x["trials"] for x in c) / max(1, sum(x["accepted"] for x in c))
    wp = [np.mean(pref[arm][p]) for p in pids if len(pref[arm].get(p, [])) == 2]
    rec["judge_win"], rec["judge_n"] = (float(np.mean(wp)) if wp else None), len(wp)
    js = [pw[arm][p] for p in pids if p in pw[arm]]
    if js:
        rec.update({"pw_n": len(js), "major_flaw": float(np.mean([j["severity"] == "major" for j in js])),
                    "flaws_per_text": float(np.mean([len(set(j["flaws"])) for j in js])), "score": float(np.mean([j["quality"] for j in js]))})
    out["arms"][arm] = rec
    print(arm, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, tuple) else v) for k, v in rec.items()}, flush=True)
json.dump(out, open(a.out, "w"), indent=1)
