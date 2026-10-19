"""Do bins of subspace keys survive paraphrase as well as bins of fully random keys?  (follow-up of basis_keys.py)

Texts: the unwatermarked Llama-3B benchmark texts (runs/ob_bench, arm base) and their light / strong paraphrases by
gemma-4-31b (runs/ob_bench_para_light, runs/ob_bench_para_strong; paired by parent_id).  Sentences of an original and
its paraphrase are paired by greedy cosine matching (scripts/embedder_study.greedy_align, cosine >= 0.5).
Keys: exactly the keys of basis_keys.py (same seeds; same within-context basis fitted on runs/ss_corpus).
Per key type and paraphrase level, averaged over its 100 keys:
  bin_keep        share of aligned pairs whose 3-bit bin is unchanged
  bin_keep_clear  the same among original sentences clear of 0.02 on all three planes (what the margin is for)
  sign_keep       per-plane sign unchanged
  move / spread   mean |projection change| along a normal vs mean |projection| of the originals
"""

import glob
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wmworld import semwm  # noqa: E402
from wmworld.records import write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("es", ROOT / "scripts" / "embedder_study.py")
es = importlib.util.module_from_spec(spec); spec.loader.exec_module(es)
spec2 = importlib.util.spec_from_file_location("bk", ROOT / "scripts" / "basis_keys.py")
bk = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(bk)

N_KEYS = bk.N_KEYS
BINW = np.array([4, 2, 1])


def main():
    rng = random.Random(0)
    fit_units = [u for r in bk.texts_of("runs/ss_corpus") for u in r["units"] if len(u["cands"]) >= 8]
    fit_units = rng.sample(fit_units, min(2000, len(fit_units)))
    E = np.load("runs/basis_cache.npz")["E"]
    assert len(E) == sum(len(u["cands"]) for u in fit_units), "cache does not match the fit sample"
    W, k = np.zeros((E.shape[1],) * 2), 0
    for u in fit_units:
        X = E[k:k + len(u["cands"])]; k += len(u["cands"]); X = X - X.mean(0); W += X.T @ X
    ev, U = np.linalg.eigh(W / len(E)); U = U[:, np.argsort(ev)[::-1]]
    orig = {}
    for f in glob.glob("runs/ob_bench/generations.base.shard*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            if g.get("text"):
                orig[g["id"]] = g["text"]
    pairs_by_level = {}
    for lvl in ("para_light", "para_strong"):
        rows = [json.loads(l) for l in open(f"runs/ob_bench_{lvl}/generations.jsonl")]
        pairs_by_level[lvl] = [(orig[r["parent_id"]], r["text"]) for r in rows if r["arm"] == "base" and r.get("text") and r["parent_id"] in orig]
    emb = semwm.Embedder(bk.EMB, device="cpu", batch_size=256)
    aligned = {}
    for lvl, pairs in pairs_by_level.items():
        so = [semwm.split_sentences(a) for a, _ in pairs]; sp = [semwm.split_sentences(b) for _, b in pairs]
        eo = emb.encode([s for ss in so for s in ss]).astype(np.float64); ep = emb.encode([s for ss in sp for s in ss]).astype(np.float64)
        A, B, io, ip = [], [], 0, 0
        for a_s, b_s in zip(so, sp):
            xa, xb = eo[io:io + len(a_s)], ep[ip:ip + len(b_s)]; io += len(a_s); ip += len(b_s)
            for i, j in es.greedy_align(xa, xb):
                A.append(xa[i]); B.append(xb[j])
        aligned[lvl] = (np.array(A), np.array(B))
        print(f"{lvl}: {len(pairs)} texts, {len(A)} aligned sentence pairs", flush=True)
    types = [("random768", None)] + [(f"sub{K}", K) for K in bk.KS]
    out = {"n_keys": N_KEYS, "n_pairs": {l: int(len(v[0])) for l, v in aligned.items()}, "types": {}}
    for ti, (name, K) in enumerate(types):
        rec = {}
        for lvl, (A, B) in aligned.items():
            m = {x: [] for x in ("bin_keep", "bin_keep_clear", "sign_keep", "move", "spread")}
            for s in range(N_KEYS):
                seed = 30000 + 1000 * ti + s                             # the keys of basis_keys.py
                g = np.random.RandomState(seed).randn(3, E.shape[1] if K is None else K)
                N = g if K is None else g @ U[:, :K].T
                N /= np.linalg.norm(N, axis=1, keepdims=True)
                pa, pb = A @ N.T, B @ N.T
                keep = ((pa > 0) @ BINW) == ((pb > 0) @ BINW)
                clear = np.min(np.abs(pa), axis=1) >= semwm.MARGIN
                m["bin_keep"].append(float(keep.mean())); m["bin_keep_clear"].append(float(keep[clear].mean()) if clear.any() else np.nan)
                m["sign_keep"].append(float(np.mean((pa > 0) == (pb > 0))))
                m["move"].append(float(np.mean(np.abs(pb - pa)))); m["spread"].append(float(np.mean(np.abs(pa))))
            rec[lvl] = {x: float(np.nanmean(v)) for x, v in m.items()}
        out["types"][name] = rec
        print(f"{name:10s} " + " | ".join(f"{l}: bin kept {r['bin_keep']:.3f}, among clear {r['bin_keep_clear']:.3f}, sign kept {r['sign_keep']:.3f}, "
                                         f"move {r['move']:.4f} / spread {r['spread']:.4f}" for l, r in rec.items()), flush=True)
    write_json("runs/basis_keys_paraphrase.json", out)


if __name__ == "__main__":
    main()
