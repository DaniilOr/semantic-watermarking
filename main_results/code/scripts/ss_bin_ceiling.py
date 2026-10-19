"""Ceiling of the bin-adapter objective, from the corpus alone (CPU).

For a perfect conditional model q_b = p(x | ctx, bin(x) = b), the validation NLL gain over the base on a sentence x
of bin b is exactly -log P(b | ctx).  P(b | ctx) is estimated leave-one-out from the OTHER candidates of the same
context (add-0.5 smoothing over the label set), so the gain is not inflated by the sentence's own label.
Reported per sentence and per token, for two label sets:
  bin      the 8 LSH bins (what adapters v1 are trained on)
  region   bin b AND clear of the 0.02 margin on all 3 planes (9 labels: 8 clear regions + "near a plane")
Also: share of candidates clearing the margin, per bin.
"""
import glob
import json
import sys
from collections import Counter

import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "runs/ss_corpus_val"
corpus = {}
for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
    for ln in open(f):
        r = json.loads(ln); corpus[r["prompt_id"]] = r
g_bin, g_reg, ntok, n_ctx, clear_by_bin, all_by_bin = [], [], [], 0, Counter(), Counter()
for c in corpus.values():
    for u in c["units"]:
        if u["mask"] is None or len(u["cands"]) < 4:
            continue
        n_ctx += 1
        bins = [cd["bin"] for cd in u["cands"]]
        clear = [min(abs(x) for x in cd["proj"]) >= 0.02 for cd in u["cands"]]
        reg = [b if cl else 8 for b, cl in zip(bins, clear)]
        for labels, out, K in ((bins, g_bin, 8), (reg, g_reg, 9)):
            cnt = Counter(labels); n = len(labels)
            for l in labels:
                out.append(-np.log((cnt[l] - 1 + 0.5) / (n - 1 + 0.5 * K)))
        ntok += [len(cd["ids"]) for cd in u["cands"]]
        for b, cl in zip(bins, clear):
            all_by_bin[b] += 1; clear_by_bin[b] += cl
mt = float(np.mean(ntok))
print(f"{run}: {n_ctx} contexts, {len(ntok)} candidates, mean {mt:.1f} tokens per candidate")
for name, g in (("bin (8 labels)", g_bin), ("region (bin & clear 0.02, 9 labels)", g_reg)):
    g = np.array(g)
    print(f"  {name:38s} ceiling gain per sentence: mean {g.mean():.3f} nats (median {np.median(g):.3f}); per token {g.mean() / mt:.4f} nats")
print("  share of candidates clearing 0.02, per bin: " + " ".join(f"{b}:{clear_by_bin[b] / all_by_bin[b]:.3f}" for b in range(8)))
print(f"  overall share clearing 0.02: {sum(clear_by_bin.values()) / sum(all_by_bin.values()):.3f}")
