"""Why PMark detects BookSum worse than RealNews: per (run, arm, domain) of a score cache, the z distribution
(median, 99th / 99.9th percentile), sentences per text, the share of channel-sentence pairs that TIE with the resampled
median (|F - median| <= delta: counted as a hit for either key bit), the hit rate on the remaining pairs, and the share of
sentences whose 64 resamples were all identical to each other (median == sentence proxy exactly).  For PMark texts also
the generator / detector sentence alignment per domain.  Read-only.
Usage: python scripts/pmark_domain_probe.py CACHE RUN_WITH_PMARK_TEXTS"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmark_arm as pm
from wmworld.stages import collect_run_texts

cache, gen_run = sys.argv[1], sys.argv[2]
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
print(f"{'run':22s} {'arm':6s} {'domain':12s} {'n':>5s} {'sent':>5s} {'z med':>6s} {'z p99':>6s} {'z p99.9':>7s} {'ties':>6s} {'hit|no tie':>10s} {'exact':>6s}")
for (run, arm, d), ks in sorted(groups.items()):
    if "attack" in run or any(x in run for x in ("pegasus", "parrot")):
        continue
    zs, ns, tie, hit, exact = [], [], [], [], []
    for k in ks:
        r = recs.get(k)
        if r is None:
            continue
        c = []
        for t in range(r["n_sent"]):
            bits = pm.key_bits(t + 1)
            c += pm.soft_counts(r["proxy"][t], r["median"][t] or [None] * 4, bits)
            if not r["median"][t]:
                continue
            ex = all(abs(cs - m) < 1e-6 for cs, m in zip(r["proxy"][t], r["median"][t]))
            exact.append(ex)
            for cs, m, b in zip(r["proxy"][t], r["median"][t], bits):
                if abs(cs - m) <= pm.DELTA:
                    tie.append(1)
                else:
                    tie.append(0); hit.append((cs > m) if b else (cs < m))
        zs.append(pm.z_of(c)); ns.append(r["n_sent"])
    if not zs:
        continue
    zs = np.array([z for z in zs if z is not None])
    print(f"{run:22s} {arm:6s} {d:12s} {len(zs):5d} {np.mean(ns):5.1f} {np.median(zs):6.2f} {np.percentile(zs, 99):6.2f} {np.percentile(zs, 99.9):7.2f} "
          f"{np.mean(tie):6.3f} {np.mean(hit):10.3f} {np.mean(exact):6.3f}")
by = defaultdict(lambda: [0, 0])
for g in collect_run_texts(Path(gen_run))[0]:
    if g["arm"] != "pmark" or g.get("status", "ok") != "ok":
        continue
    r = recs.get(f"{g['text_hash']}|{g['prompt_id']}"); tr = (g.get("cost") or {}).get("sentence_trace")
    if r is None or tr is None:
        continue
    by[g["domain"]][0] += 1; by[g["domain"]][1] += r["n_sent"] != len(tr)
for d, (n, bad) in sorted(by.items()):
    print(f"alignment {d}: {bad} of {n} PMark texts split differently at detection")
