"""How often do logged candidates land in their mask (and clear 0.02)?  Pushed corpus vs plain corpus, from the logged
projections only (no model, no embedding).  Unit 0 (the plain seed sentence) is excluded; so are exact duplicates' effect
(reported separately)."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld import semwm
pl = semwm.Hyperplanes(np.eye(3))
for run in sys.argv[1:]:
    rows = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); rows[r["prompt_id"]] = r
    n = hit = clr = dup = units = 0; per_unit_any = []
    for r in rows.values():
        for u in r["units"]:
            if u["mask"] is None:
                continue
            units += 1; P = [np.array(c["proj"]) for c in u["cands"]]
            d = [semwm.signed_distance(q, tuple(u["mask"]), pl) for q in P]
            n += len(P); hit += sum(x > 0 for x in d); clr += sum(x > 0 and np.min(np.abs(q)) >= 0.02 for x, q in zip(d, P))
            dup += len(P) - len({tuple(q) for q in (tuple(c["proj"]) for c in u["cands"])})
            per_unit_any.append(any(x > 0 and np.min(np.abs(q)) >= 0.02 for x, q in zip(d, P)))
    print(f"{run}: texts {len(rows)}, positions {units}, candidates {n} | in mask {hit / n:.3f} | in mask & clear 0.02 {clr / n:.3f} "
          f"| positions with >=1 clear candidate {np.mean(per_unit_any):.3f} | duplicate candidates {dup / n:.3f}")
