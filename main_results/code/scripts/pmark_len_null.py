"""Length check for PMark's human null: token lengths (Llama-3.2 tokenizer) of the human null texts vs the PMark and plain
texts, the correlation of the null z with the number of sentences, and PMark TPR when the human null is cut to the same
200-token budget as the generated texts (z over the sentences that start within the first 200 tokens; their cached
proxies / medians are exactly what a cut text gets, except the last sentence is kept whole).  Read-only.
Usage: python scripts/pmark_len_null.py CACHE NULL_RUN_NAME PMARK_RUN_NAME PLAIN_RUN_NAME BUDGET DOMAIN [DOMAIN ...]"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmark_arm as pm
from wmworld.posstats import upper_tail_threshold, wilson
from wmworld.stages import collect_run_texts

cache, null_run, pm_run, plain_run, budget, doms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6:]
tok = AutoTokenizer.from_pretrained(json.load(open("assets/ss_models.json"))["llama3b"]["path"])
recs = {}
for f in glob.glob(f"{cache}/scores.shard*.jsonl"):
    for l in open(f):
        try:
            r = json.loads(l); recs[r["key"]] = r
        except json.JSONDecodeError:
            pass

def z_first(r, m):
    c = []
    for t in range(min(m, r["n_sent"])):
        c += pm.soft_counts(r["proxy"][t], r["median"][t] or [None] * 4, pm.key_bits(t + 1))
    return pm.z_of(c)

texts = defaultdict(list)
for run in (null_run, pm_run, plain_run):
    for g in collect_run_texts(Path("runs") / run)[0]:
        if g.get("status", "ok") == "ok" and g.get("text") and g["domain"] in doms and g["arm"] in ("base", "pmark"):
            if (run == pm_run) == (g["arm"] == "pmark"):
                texts[(run, g["domain"])].append(g)
for d in doms:
    out = {}
    for run in (null_run, pm_run, plain_run):
        L, Z, Zcut, N, Ncut = [], [], [], [], []
        for g in texts[(run, d)]:
            r = recs.get(f"{g['text_hash']}|{g['prompt_id']}")
            if r is None:
                continue
            text = pm.semwm.normalize_text(g["text"])
            offs = pm.sentence_offsets(text)
            starts = [len(tok(text[:o], add_special_tokens=False)["input_ids"]) for o, _ in offs]
            m = sum(s < budget for s in starts)
            L.append(len(tok(text, add_special_tokens=False)["input_ids"])); N.append(r["n_sent"]); Ncut.append(m)
            Z.append(z_first(r, r["n_sent"])); Zcut.append(z_first(r, m))
        out[run] = (np.array(L), np.array(Z, dtype=float), np.array(Zcut, dtype=float), np.array(N), np.array(Ncut))
        print(f"{d:12s} {run:22s} n={len(L):5d} tokens median {np.median(L):6.0f} p90 {np.percentile(L, 90):6.0f} | sentences {np.mean(N):5.1f} "
              f"(first {budget} tokens: {np.mean(Ncut):5.1f}) | corr(z, sentences) {np.corrcoef(Z, N)[0, 1]:+.2f}")
    hz, hzc = out[null_run][1], out[null_run][2]
    for name, h in (("human null as is", hz), (f"human null cut to {budget} tokens", hzc)):
        t1, t01 = upper_tail_threshold(list(h), 0.01), upper_tail_threshold(list(h), 0.001)
        line = f"{d:12s} {name:34s} thr1% {t1:5.2f} thr0.1% {t01:5.2f} |"
        for run, lab in ((pm_run, "PMark"), (plain_run, "plain")):
            z = out[run][1]
            k1, k01 = int(np.sum(z > t1)), int(np.sum(z > t01))
            line += f" {lab} {100 * k1 / len(z):5.1f} / {100 * k01 / len(z):5.1f} |"
        print(line)
