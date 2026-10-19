"""Why human BookSum text scores high under the k-means partition: per partition, on the human null texts of a domain,
share of consecutive sentence pairs in the same region and share of sentences 2..n inside the previous one's mask
(25% expected by chance).  Read-only.  Usage: python scripts/ks_null_check.py DOMAIN PARTITION(c4|booksum|lsh)"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import ksemstamp as K
from wmworld import semwm
dom, part = sys.argv[1], sys.argv[2]
enc = K.ENCODERS["c4"] if part in ("c4", "lsh") else K.ENCODERS[part]
if part != "lsh":
    K.install(K.CENTROIDS[part])
emb = semwm.Embedder(enc, device="cpu", batch_size=256)
key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
texts = [r["text"] for r in map(json.loads, open("runs/sp_bench_null/generations.jsonl")) if r.get("domain") == dom][:1000]
same = pairs = inm = 0; occ = np.zeros(8)
for t in texts:
    s = semwm.split_sentences(t)
    if len(s) < 2:
        continue
    b = [planes.bin_of(p) for p in planes.projections(emb.encode(s))]
    for x in b: occ[x] += 1
    for i in range(1, len(b)):
        pairs += 1; same += b[i] == b[i - 1]; inm += b[i] in semwm.valid_mask(key, b[i - 1], 0.25)
print(f"{dom} {part}: texts {len(texts)} pairs {pairs} same-region {100*same/pairs:.1f}% in-mask {100*inm/pairs:.1f}% (chance 25%) "
      f"region shares {np.round(100*occ/occ.sum(),1).tolist()}")
