"""Per-draw hit rates from the FIRST block of 4 candidates of every unit (every unit draws it, so the rate is not
biased by the stop rule): share of candidates in the unit's mask, and in the mask and clear of 0.02."""
import glob, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from wmworld import semwm
key, pl = semwm.Key.shipped(), semwm.Hyperplanes(np.eye(3))
arms = sys.argv[2:]
for arm in arms:
    by = defaultdict(lambda: [0, 0, 0])
    for f in glob.glob(f"{sys.argv[1]}/generations.{arm}.shard*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            for u in g["cost"].get("units", []):
                if u.get("prev_bin") is None:
                    continue
                m = semwm.valid_mask(key, u["prev_bin"])
                for c in u["cands"][:4]:
                    q = np.array(c[:3]); dep = semwm.signed_distance(q, m, pl)
                    s = by[g["domain"]]; s[0] += 1; s[1] += dep > 0; s[2] += dep > 0 and np.min(np.abs(q)) >= 0.02
    for d, (n, h, c) in sorted(by.items()):
        print(arm, d, "first-block draws", n, "in mask %.3f" % (h / n), "in mask & clear %.3f" % (c / n))
