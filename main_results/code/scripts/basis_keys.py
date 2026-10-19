"""Keys drawn INSIDE the top-K within-context subspace (the follow-up of scripts/basis_feasibility.py).

If every plane normal lies in span(U_K) (U_K = top-K eigenvectors of the within-position covariance of candidate
embeddings, fitted on runs/ss_corpus), a K-dim basis covers every key exactly.  Question: are such keys any good?
Per key type (fully random 768-dim normals, and normals isotropic inside span(U_K) for several K), 100 keys each:
  selection  held-out positions (runs/ss_corpus_val, 60 texts): share where the deepest of the position's candidates
             (exact key) is in the mask / in the mask and clear of 0.02; the first candidate (a random draw) as the floor
  null       unwatermarked Llama-3B texts (runs/ob_bench base, runs/ss_llama3b base): in-mask rate of consecutive
             sentences, chain z per text (mean, 99th percentile over texts and keys), bin balance (min/max bin share),
             share of sentences clear of 0.02 from all three planes, |n . mu| / sd(n . e) (plane offset)
Embeddings are cached in runs/basis_cache.npz.
"""

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.records import write_json  # noqa: E402

EMB = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
KS = [8, 16, 32, 64, 128, 256]
N_KEYS = 100
SIGNS = np.array([[1.0 if (b >> (2 - j)) & 1 else -1.0 for j in range(3)] for b in range(8)])


def texts_of(run):
    rows = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); rows[r["prompt_id"]] = r
    return list(rows.values())


def null_texts():
    out = []
    for f in glob.glob("runs/ob_bench/generations.base.shard*.jsonl") + glob.glob("runs/ss_llama3b/generations.base.shard*.jsonl"):
        for ln in open(f):
            g = json.loads(ln)
            if g.get("text") and g.get("domain") != "ob_c4":
                out.append(g["text"])
    return out


def main():
    rng = random.Random(0)
    cache = Path("runs/basis_cache.npz")
    fit_units = [u for r in texts_of("runs/ss_corpus") for u in r["units"] if len(u["cands"]) >= 8]
    fit_units = rng.sample(fit_units, min(2000, len(fit_units)))
    val = rng.sample(texts_of("runs/ss_corpus_val"), 60)
    nulls = null_texts()
    null_sents = [semwm.split_sentences(t) for t in nulls]
    if cache.exists():
        z = np.load(cache); E, V, S = z["E"], z["V"], z["S"]
    else:
        emb = semwm.Embedder(EMB, device="cpu", batch_size=256)
        E = emb.encode([c["text"] for u in fit_units for c in u["cands"]]).astype(np.float64)
        V = emb.encode([c["text"] for r in val for u in r["units"] for c in u["cands"]]).astype(np.float64)
        S = emb.encode([s for ss in null_sents for s in ss]).astype(np.float64)
        np.savez(cache, E=E, V=V, S=S)
    print(f"fit candidates {len(E)}, held-out candidates {len(V)}, null texts {len(nulls)} ({len(S)} sentences)", flush=True)
    mu = E.mean(0)
    W, k = np.zeros((E.shape[1],) * 2), 0
    for u in fit_units:
        X = E[k:k + len(u["cands"])]; k += len(u["cands"]); X = X - X.mean(0); W += X.T @ X
    ev, U = np.linalg.eigh(W / len(E)); U = U[:, np.argsort(ev)[::-1]]
    pos, k = [], 0
    for r in val:
        t = []
        for u in r["units"]:
            t.append((V[k:k + len(u["cands"])], u["chosen"])); k += len(u["cands"])
        pos.append(t)
    nt, k = [], 0
    for ss in null_sents:
        nt.append(S[k:k + len(ss)]); k += len(ss)
    types = [("random768", None)] + [(f"sub{K}", K) for K in KS]
    out = {"n_keys": N_KEYS, "types": {}}
    for ti, (name, K) in enumerate(types):
        acc = {m: [] for m in ("sel_in", "sel_clear", "rand_in", "rand_clear", "null_in", "null_z_mean", "null_z_p99",
                               "bin_balance", "margin_share", "offset")}
        zs_all = []
        for s in range(N_KEYS):
            seed = 30000 + 1000 * ti + s
            g = np.random.RandomState(seed).randn(3, E.shape[1] if K is None else K)
            N = g if K is None else g @ U[:, :K].T
            N /= np.linalg.norm(N, axis=1, keepdims=True)
            key = semwm.Key(rand_seed=seed, hash_key=semwm.HASH_KEY + 2 * seed)
            masks = [semwm.valid_mask(key, b) for b in range(8)]
            binw = np.array([4, 2, 1])
            sel = [0, 0]; rnd = [0, 0]; n_pos = 0
            for t in pos:
                for i in range(1, len(t)):
                    Xp, cp = t[i - 1]; pb = int(((Xp[cp] @ N.T) > 0) @ binw)
                    P = t[i][0] @ N.T
                    dep = np.max(np.stack([np.min(SIGNS[b] * P, axis=1) for b in masks[pb]]), axis=0)
                    clr = (dep > 0) & (np.min(np.abs(P), axis=1) >= semwm.MARGIN)
                    j = int(np.argmax(dep)); n_pos += 1
                    sel[0] += dep[j] > 0; sel[1] += clr[j]; rnd[0] += dep[0] > 0; rnd[1] += clr[0]
            ins, zs, allb = [], [], []
            for X in nt:
                if len(X) < 2:
                    continue
                b = ((X @ N.T) > 0) @ binw; allb.extend(b.tolist())
                hit = [b[i] in masks[b[i - 1]] for i in range(1, len(b))]
                ins.extend(hit); n = len(hit)
                zs.append((sum(hit) - 0.25 * n) / np.sqrt(n * 0.25 * 0.75))
            Pn = S @ N.T
            freq = np.bincount(np.array(allb), minlength=8) / len(allb)
            acc["sel_in"].append(sel[0] / n_pos); acc["sel_clear"].append(sel[1] / n_pos)
            acc["rand_in"].append(rnd[0] / n_pos); acc["rand_clear"].append(rnd[1] / n_pos)
            acc["null_in"].append(float(np.mean(ins))); acc["null_z_mean"].append(float(np.mean(zs)))
            acc["null_z_p99"].append(float(np.percentile(zs, 99))); zs_all.extend(zs)
            acc["bin_balance"].append(float(freq.min() / freq.max()))
            acc["margin_share"].append(float(np.mean(np.min(np.abs(Pn), axis=1) >= semwm.MARGIN)))
            acc["offset"].append(float(np.mean(np.abs(mu @ N.T) / (S @ N.T).std(0))))
        rec = {m: {"mean": float(np.mean(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))} for m, v in acc.items()}
        rec["null_z_p99_pooled"] = float(np.percentile(zs_all, 99))
        out["types"][name] = rec
        print(f"{name:10s} select in {rec['sel_in']['mean']:.3f} clear {rec['sel_clear']['mean']:.3f} | random draw in {rec['rand_in']['mean']:.3f} "
              f"clear {rec['rand_clear']['mean']:.3f} | null in-mask {rec['null_in']['mean']:.3f} [{rec['null_in']['p5']:.3f}-{rec['null_in']['p95']:.3f}] "
              f"z mean {rec['null_z_mean']['mean']:+.2f} z p99 pooled {rec['null_z_p99_pooled']:.2f} | bin balance {rec['bin_balance']['mean']:.2f} "
              f"margin share {rec['margin_share']['mean']:.3f} offset {rec['offset']['mean']:.2f}", flush=True)
    write_json("runs/basis_keys.json", out)


if __name__ == "__main__":
    main()
