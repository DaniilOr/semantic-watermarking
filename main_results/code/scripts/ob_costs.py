"""Generation cost per arm (and per dataset) of the OpenStamp-style benchmark run, from the records only.

chain arms: candidates drawn per unit (the first unit, a plain seed sample, excluded), share of units at the cap of 16,
kept unit in its mask / in its mask and clear of 0.02; SemStamp: candidates per accepted sentence and the share forced
through at the trial cap; every arm: wall seconds per text, tokens, words, empty texts.
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/ob_bench"
DOMAINS = ("ob_arxiv", "ob_booksum", "ob_wikipedia")


def summary(gs):
    ok = [g for g in gs if g.get("text")]
    c = [g["cost"] for g in ok]
    r = {"n": len(gs), "empty": len(gs) - len(ok),
         "s_per_text": float(np.mean([x.get("wall_seconds") or 0.0 for x in c])) if c else None,
         "tokens": float(np.mean([g["visible_tokens"] for g in ok])) if ok else None,
         "words": float(np.mean([len(g["text"].split()) for g in ok])) if ok else None}
    U = [u for x in c for u in x.get("units", []) if u.get("prev_bin") is not None and "n_drawn" in u]
    if U:
        nd = np.array([u["n_drawn"] for u in U])
        kept = [u["cands"][u["chosen"]][:3] for u in U]
        r.update({"units": len(U), "cand_per_unit": float(nd.mean()), "cand_median": float(np.median(nd)),
                  "share_at_16": float(np.mean(nd >= 16)), "kept_in_mask": float(np.mean([u["in_mask"] for u in U])),
                  "kept_clear_0.02": float(np.mean([u["in_mask"] and min(abs(v) for v in q) >= 0.02 for u, q in zip(U, kept)]))})
    if any("trials" in x for x in c):
        tr, acc, mx = (sum(x.get(k, 0) for x in c) for k in ("trials", "accepted", "maxed_out"))
        r.update({"cand_per_sentence": tr / max(1, acc), "forced_share": mx / max(1, acc), "sentences": acc})
    return r


by = defaultdict(list)
for f in sorted(glob.glob(f"{RUN}/generations.*.shard*.jsonl")):
    for ln in open(f):
        g = json.loads(ln)
        if g.get("domain") in DOMAINS:
            by[g["arm"]].append(g)
out = {}
for arm, gs in sorted(by.items()):
    out[arm] = {"all": summary(gs)} | {d: summary([g for g in gs if g["domain"] == d]) for d in DOMAINS}
    for k, r in out[arm].items():
        print(arm, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
json.dump(out, open(f"{RUN.rstrip('/')}_costs.json", "w"), indent=1)
