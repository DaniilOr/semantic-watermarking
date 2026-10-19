"""Is the judged damage tied to the push on the OPENING of a text or to the total push?  (existing traces + pointwise judgments)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.posstats import auroc  # noqa: E402

SLUG = "google_gemma-4-31b-it"
for run, arm in (("conf_cross", "wm_c4"), ("conf_cross", "wm_cross30"), ("keylen10", "wm_c4")):
    rd = Path("runs") / run
    gens = {}
    for f in rd.glob("generations*.jsonl"):
        for ln in open(f):
            g = json.loads(ln); gens[g["id"]] = g
    J = {}
    for ln in open(rd / "eval" / f"api_pointwise_{SLUG}.jsonl"):
        r = json.loads(ln); J[(r["prompt_id"], r["arm"])] = r["judgment"]
    rows = []
    for f in rd.glob("trajectories*.jsonl"):
        for ln in open(f):
            if '"per_token"' not in ln or f".{arm}." not in ln[:300]:
                continue
            r = json.loads(ln); g = gens.get(r["output_id"])
            if not g or (g["prompt_id"], arm) not in J or (g["prompt_id"], "base") not in J:
                continue
            kl = np.asarray(r["cost"]["per_token"]["kl"], dtype=np.float64)
            if kl.size < 96:
                continue
            jm, jc = J[(g["prompt_id"], arm)], J[(g["prompt_id"], "base")]
            bad = lambda j: bool({"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"]))
            rows.append({"open": kl[:64].sum(), "rest": kl[64:].sum(), "rest_rate": kl[64:].mean(), "total": kl.sum(), "flag": bad(jm),
                         "dq": jm["quality"] - jc["quality"], "off": jm["addresses_task"] != "yes"})
    F = [r for r in rows if r["flag"]]; U = [r for r in rows if not r["flag"]]
    dq = np.array([r["dq"] for r in rows])
    print(f"\n{run}/{arm}: {len(rows)} texts with clean counterpart judged, {len(F)} flagged for insertions/drift; mean quality change {dq.mean():+.2f}")
    print(f"   {'push measure':34s} {'flagged':>8s} {'unflagged':>9s} {'AUROC(flag)':>11s} {'corr with quality change':>25s}")
    for k, label in (("open", "KL spent on the first 64 tokens"), ("rest", "KL spent after token 64"), ("rest_rate", "KL per token after token 64"), ("total", "total KL")):
        x = np.array([r[k] for r in rows])
        print(f"   {label:34s} {np.mean([r[k] for r in F]):8.2f} {np.mean([r[k] for r in U]):9.2f} {auroc([r[k] for r in F], [r[k] for r in U]):11.2f} {np.corrcoef(x, dq)[0, 1]:+25.2f}")
