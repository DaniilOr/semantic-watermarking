"""Print the OpenStamp-style benchmark detection table from runs/ob_detect*.json (per dataset; TPR at the human-null
1% / 0.1% FPR thresholds and the model-null 1% threshold; AUROC vs human; flag rates of the unwatermarked cohorts)."""

import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "runs/ob_detect.json"
D = ["ob_arxiv", "ob_booksum", "ob_wikipedia"]
BASE = sys.argv[2] if len(sys.argv) > 2 else "ob_bench"                  # run name, e.g. ob_bench_llama1b
RUNS = [BASE] + [BASE + x for x in ("_reorder_sent", "_resegment", "_para_light", "_para_strong")]
OWN = {"semstamp": "z_chain", "chain_reference": "z_chain", "chain_grpo300": "z_chain", "chain_grpo400": "z_chain",
       "chain_rft3": "z_chain", "chain_seg_push": "z_segchain", "chain_seg_push_m": "z_segchain"}


def pct(x):
    return "-" if x is None else "%.1f" % (100 * x["rate"])


d = json.load(open(PATH))
R = d["results"]
D = sorted({k.split("|")[2] for k in R})                                # the datasets present in this file
arms = sorted({k.split("|")[1] for k in R})
print("TPR %: human-null 1% / human-null 0.1% / model-null 1%   (runs: clean, reorder, resegment, para-light, para-strong)")
for arm in [a for a in OWN if a in arms] + sorted(a for a in arms if a.startswith("chain_") and a not in OWN):
    st = OWN.get(arm, "z_segchain" if arm.startswith("chain_seg") else "z_chain")
    for dm in D:
        cells = []
        for r in RUNS:
            x = R.get("%s|%s|%s" % (r, arm, dm))
            if not x:
                cells.append("-")
                continue
            c = "%s/%s/%s n%d" % (pct(x[st + ":human_1%"]), pct(x[st + ":human_0.1%"]), pct(x[st + ":model_1%"]), x["n"])
            if r == BASE:
                c += " auroc %.3f" % x[st + ":auroc_vs_human"]
            cells.append(c)
        print(arm, st, dm, " | ".join(cells))
print("\nflag rate % of unwatermarked cohorts (clean run): human-null 1% / 0.1%")
for arm in ["base", "clean_seed1", "human"]:
    for st in ["z_chain", "z_segchain"]:
        out = []
        for dm in D:
            x = R.get("%s|%s|%s" % (BASE, arm, dm))
            if x:
                out.append("%s %s/%s n%d" % (dm, pct(x[st + ":human_1%"]), pct(x[st + ":human_0.1%"]), x["n"]))
        print(arm, st, " | ".join(out))
print("\nthresholds")
for k, v in d["thresholds"].items():
    if k.split(":")[1] in ("z_chain", "z_segchain"):
        print(k, {t: (round(v[t], 3) if isinstance(v[t], float) else v[t]) for t in ("human_1%", "human_0.1%", "model_1%", "n_human", "n_model")})
