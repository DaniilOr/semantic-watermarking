"""Push + margin (arm chain_reference) cost on the 12 columns of docs/semstamp_slide.tex: per sentence after the first,
mean candidates drawn, share kept outside the mask (no usable candidate among 16), share kept in the mask but inside
the 0.02 margin; mean wall seconds per text.  Read-only; prints JSON and writes runs/push_cost_cols.json."""

import glob
import json
import statistics as St
import sys

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from semstamp_slide import runs  # noqa: E402
from semstamp_tables import OB, SP  # noqa: E402

out = {}
for bench, doms in (("sp", SP), ("ob", OB)):
    for kind, short, run, _, _ in runs(bench):
        drawn = units = outm = thin = 0; ws = []
        for f in glob.glob(f"runs/{run}/generations.chain_reference*.jsonl"):
            for l in open(f, encoding="utf-8"):
                if not l.strip().endswith("}"):
                    continue
                r = json.loads(l)
                if r.get("domain") not in doms or r.get("status", "ok") != "ok":
                    continue
                c = r.get("cost") or {}
                ws.append(c.get("wall_seconds", 0))
                for u in (c.get("units") or [])[1:]:
                    units += 1; drawn += u["n_drawn"]
                    if not u["in_mask"]:
                        outm += 1
                    elif (u.get("depth") or 0) < 0.02:
                        thin += 1
        out[f"{bench}|{kind}|{short}"] = {"run": run, "units": units, "cand": drawn / units if units else None,
                                          "out_of_mask": 100 * outm / units if units else None,
                                          "in_mask_inside_margin": 100 * thin / units if units else None,
                                          "sec": St.mean(ws) if ws else None, "texts": len(ws)}
json.dump(out, open("runs/push_cost_cols.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
