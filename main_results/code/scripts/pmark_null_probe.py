"""Why does unwatermarked text score high under PMark's soft z-test?  Per (run, arm, domain) group of a score cache:
mean z with the released soft count (K=250), with the paper's K=150, and with hard counts only (a miss counts 0);
the share of channel-sentence pairs on the key's side ("hard hits", 0.5 expected without a watermark); and the mean
soft credit a miss receives, exp(-K |F - median|).  Read-only.
Usage: python scripts/pmark_null_probe.py CACHE_DIR [CACHE_DIR ...]"""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmark_arm as pm

for cache in sys.argv[1:]:
    recs = {}
    for f in glob.glob(f"{cache}/scores.shard*.jsonl"):
        for l in open(f):
            r = json.loads(l); recs[r["key"]] = r
    groups = defaultdict(set)
    for f in glob.glob(f"{cache}/groups*.json"):
        for g in json.load(open(f)):
            groups[(g["run"], g["arm"], g["domain"])].add(g["key"])
    print(f"== {cache}")
    for (run, arm, d), ks in sorted(groups.items()):
        zs = {250: [], 150: [], 0: []}; hits = []; credit = []; nsent = []; spread = []
        for k in ks:
            r = recs.get(k)
            if r is None:
                continue
            nsent.append(r["n_sent"])
            for K in zs:
                c = []
                for t in range(r["n_sent"]):
                    c += pm.soft_counts(r["proxy"][t], r["median"][t] or [None] * 4, pm.key_bits(t + 1), K if K else None)
                zs[K].append(pm.z_of(c))
            for t in range(r["n_sent"]):
                if not r["median"][t]:
                    continue
                for cs, m, b in zip(r["proxy"][t], r["median"][t], pm.key_bits(t + 1)):
                    hit = (cs >= m - pm.DELTA) if b else (cs <= m + pm.DELTA)
                    hits.append(hit)
                    if not hit:
                        credit.append(math.exp(-250 * abs(cs - m)))
                    spread.append(abs(cs - m))
        if not nsent:
            continue
        f = lambda v: f"{np.mean(v):6.2f}" if v else "   -  "
        print(f"{run:24s} {arm:6s} {d:18s} n={len(nsent):2d} sent={np.mean(nsent):5.1f} | z K=250 {f(zs[250])}  K=150 {f(zs[150])}  hard {f(zs[0])} | "
              f"hard hits {np.mean(hits):.3f}  miss credit {np.mean(credit) if credit else float('nan'):.3f}  |F-med| {np.median(spread):.4f}")
