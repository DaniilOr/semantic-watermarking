"""Diagnostic (NOT PMark's rule): z with a neutral count, tie (|F - median| <= delta) = 0.5 and miss = 0, which has mean
0.5 without a watermark whatever the number of sentences; compared with PMark's released soft count.  Thresholds on the
human null at 1% / 0.1%; TPR of PMark texts and of the plain model's texts.  Read-only.
Usage: python scripts/pmark_neutral_ties.py CACHE NULL_RUN PMARK_RUN PLAIN_RUN DOMAIN [DOMAIN ...]"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmark_arm as pm
from wmworld.posstats import upper_tail_threshold

cache, null_run, pm_run, plain_run, doms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
recs = {}
for f in glob.glob(f"{cache}/scores.shard*.jsonl"):
    for l in open(f):
        try:
            r = json.loads(l); recs[r["key"]] = r
        except json.JSONDecodeError:
            pass
groups = defaultdict(set)
for f in glob.glob(f"{cache}/groups*.json"):
    for g in json.load(open(f)):
        groups[(g["run"], g["arm"], g["domain"])].add(g["key"])

def z(r, neutral):
    c = []
    for t in range(r["n_sent"]):
        med, bits = r["median"][t], pm.key_bits(t + 1)
        if not med:
            continue
        if not neutral:
            c += pm.soft_counts(r["proxy"][t], med, bits)
        else:
            for cs, m, b in zip(r["proxy"][t], med, bits):
                c.append(0.5 if abs(cs - m) <= pm.DELTA else float((cs > m) if b else (cs < m)))
    return pm.z_of(c)

for d in doms:
    for neutral, name in ((False, "PMark soft count (released)"), (True, "neutral ties, hard misses")):
        val = {k: [z(recs[x], neutral) for x in ks if x in recs] for k, ks in
               {"null": groups[(null_run, "base", d)], "pmark": groups[(pm_run, "pmark", d)], "plain": groups[(plain_run, "base", d)]}.items()}
        t1, t01 = upper_tail_threshold(val["null"], 0.01), upper_tail_threshold(val["null"], 0.001)
        f = lambda v, t: 100 * np.mean(np.array(v) > t)
        print(f"{d:12s} {name:28s} human z p99.9 {np.percentile(val['null'], 99.9):5.2f} thr {t1:5.2f}/{t01:5.2f} | "
              f"PMark {f(val['pmark'], t1):5.1f} / {f(val['pmark'], t01):5.1f} | plain {f(val['plain'], t1):4.1f} / {f(val['plain'], t01):4.1f}")
