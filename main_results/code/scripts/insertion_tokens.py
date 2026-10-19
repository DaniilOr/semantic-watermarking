"""Are the judge's insertion / drift flags tied to chosen tokens the key makes MUCH more likely (raw contrast > 1)?"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.posstats import auroc  # noqa: E402

SLUG = "google_gemma-4-31b-it"
for run, arm in (("conf_window_reg", "wm_w64_c3"), ("conf_cross", "wm_c4"), ("conf_cross", "wm_cross30")):
    rd = Path("runs") / run
    gens = {json.loads(l)["id"]: json.loads(l) for f in rd.glob("generations*.jsonl") for l in open(f)}
    J = {json.loads(l)["gen_id"]: json.loads(l)["judgment"] for l in open(rd / "eval" / f"api_pointwise_{SLUG}.jsonl")}
    rows = []
    for f in rd.glob("trajectories*.jsonl"):
        for ln in open(f):
            if '"per_token"' not in ln or f".{arm}." not in ln[:300]:
                continue
            r = json.loads(ln)
            if r["output_id"] not in J:
                continue
            pt = r["cost"]["per_token"]
            raw, beta = np.asarray(pt["contrast_raw"]), np.asarray(pt["beta_t"])
            n = raw.size
            if n < 64:
                continue
            j = J[r["output_id"]]
            rows.append({"flag": bool({"unrelated_words_or_insertions", "drifts_off_topic"} & set(j["flaws"])), "q": j["quality"],
                         "hi1": 100.0 * ((raw > 1.0) & (beta > 0)).sum() / n, "hi2": 100.0 * ((raw > 2.0) & (beta > 0)).sum() / n,
                         "hi1_open": float(((raw[:64] > 1.0) & (beta[:64] > 0)).sum()), "mid": 100.0 * ((raw > 0.1) & (raw <= 1.0) & (beta > 0)).sum() / n,
                         "share_ev_hi": float(np.clip(raw[raw > 1.0], -0.5, 0.5).sum() / max(1e-9, np.clip(raw, -0.5, 0.5).sum()))})
    F = [x for x in rows if x["flag"]]; U = [x for x in rows if not x["flag"]]
    print(f"\n{run}/{arm}: {len(rows)} texts, {len(F)} flagged for insertions or drift")
    for k, label in (("hi1", "pushed tokens with raw contrast > 1, per 100 tokens"), ("hi2", "pushed tokens with raw contrast > 2, per 100 tokens"),
                     ("hi1_open", "such tokens (> 1) among the first 64"), ("mid", "pushed tokens with raw contrast 0.1-1, per 100 tokens")):
        q = np.array([x["q"] for x in rows]); v = np.array([x[k] for x in rows])
        print(f"   {label:54s} flagged {np.mean([x[k] for x in F]):6.2f}  not flagged {np.mean([x[k] for x in U]):6.2f}  AUROC {auroc([x[k] for x in F], [x[k] for x in U]):.2f}  corr with score {np.corrcoef(q, v)[0, 1]:+.2f}")
    print(f"   share of a text's clipped evidence carried by tokens with raw contrast > 1: mean {np.mean([x['share_ev_hi'] for x in rows]):.2f}")
