"""Zero-generation feasibility of the "basis" idea: can a fixed K-dim subspace of the encoder's space serve ANY key?

Data: the logged candidate groups (up to 16 exact Llama-3B samples per sentence position) of runs/ss_corpus (fit)
and runs/ss_corpus_val (held-out prompts).  Basis = top-K eigenvectors of the WITHIN-CONTEXT covariance of the
candidates' SemStamp embeddings, i.e. the directions in which the generator's options at one position differ.
A key's projection is approximated as  p_hat = n . mu + (P_K n) . (e - mu)  (mu = fit-set mean: the key-specific
offset is computed exactly, only the varying part is restricted to the basis).

Per K, on held-out data:
  within_r2    R^2 between the within-position deviations of p_hat and of the exact projection n . e, averaged over
               1,002 random plane normals (mean and 5th percentile): how well the basis ranks a position's options
  sign_agree   P[sign(p_hat) == sign(n . e)] over held-out candidates and the same normals
  select       200 random keys (3 planes + their own mask stream) x every held-out position after the first: keep the
               candidate deepest in the mask by p_hat; share of positions whose kept candidate is truly in the mask /
               in the mask and clear of 0.02.  Bounds: exact-key selection (upper) and the first candidate (a random
               draw, lower).  The shipped key is reported on its own.
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
KS = [4, 8, 16, 32, 64, 128, 256, 768]
N_FIT_UNITS, N_VAL_TEXTS, N_KEYS = 2000, 60, 200


def texts_of(run):
    out = []
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            out.append(json.loads(ln))
    return {r["prompt_id"]: r for r in out}.values()                    # last line per prompt wins


def main():
    rng = random.Random(0)
    emb = semwm.Embedder(EMB, device="cpu", batch_size=256)
    # ---- fit set: within-context covariance ---------------------------------------------------- #
    fit_units = [u for r in texts_of("runs/ss_corpus") for u in r["units"] if len(u["cands"]) >= 8]
    fit_units = rng.sample(fit_units, min(N_FIT_UNITS, len(fit_units)))
    E = emb.encode([c["text"] for u in fit_units for c in u["cands"]]).astype(np.float64)
    mu = E.mean(0)
    W, k = np.zeros((E.shape[1], E.shape[1])), 0
    for u in fit_units:
        X = E[k:k + len(u["cands"])]; k += len(u["cands"])
        X = X - X.mean(0); W += X.T @ X
    W /= len(E)
    evals, evecs = np.linalg.eigh(W)
    order = np.argsort(evals)[::-1]; evals, evecs = evals[order], evecs[:, order]
    print(f"fit: {len(fit_units)} positions, {len(E)} candidates; within-context variance in top 8/32/128 eigvecs: "
          + " / ".join(f"{evals[:k_].sum() / evals.sum():.3f}" for k_ in (8, 32, 128)), flush=True)
    # ---- held-out texts: every position, with its candidates ----------------------------------- #
    val = rng.sample(list(texts_of("runs/ss_corpus_val")), N_VAL_TEXTS)
    V = emb.encode([c["text"] for r in val for u in r["units"] for c in u["cands"]]).astype(np.float64)
    pos, k = [], 0                                                          # per text: list of (cands [n,768], chosen)
    for r in val:
        units = []
        for u in r["units"]:
            units.append((V[k:k + len(u["cands"])], u["chosen"])); k += len(u["cands"])
        pos.append(units)
    Vc = V - mu
    within = [(X - X.mean(0)) for t in pos for X, _ in t if len(X) >= 8]
    Wv = np.vstack(within)
    normals = np.vstack([np.random.RandomState(10000 + s).randn(3, E.shape[1]) for s in range(334)])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    out = {"n_fit_positions": len(fit_units), "n_fit_candidates": int(len(E)), "n_val_texts": N_VAL_TEXTS,
           "n_val_candidates": int(len(V)), "eig_share": {k_: float(evals[:k_].sum() / evals.sum()) for k_ in KS}, "K": {}}
    exact_V = V @ normals.T; exact_W = Wv @ normals.T
    for K in KS:
        P = evecs[:, :K]
        nK = normals @ P @ P.T                                              # [n_normals, 768]
        approx_W = Wv @ nK.T
        r2 = np.array([np.corrcoef(exact_W[:, i], approx_W[:, i])[0, 1] ** 2 for i in range(len(normals))])
        approx_V = (mu @ normals.T)[None, :] + Vc @ nK.T
        agree = float(np.mean(np.sign(approx_V) == np.sign(exact_V)))
        out["K"][K] = {"within_r2_mean": float(r2.mean()), "within_r2_p5": float(np.percentile(r2, 5)), "sign_agree": agree}
    # ---- selection with the approximated key -------------------------------------------------- #
    keys = [semwm.Key.shipped()] + [semwm.Key(rand_seed=20000 + s, hash_key=semwm.HASH_KEY + 2 * (20000 + s)) for s in range(N_KEYS)]
    sel = {K: {"in": [], "clear": []} for K in KS}
    bounds = {"exact": {"in": [], "clear": []}, "random": {"in": [], "clear": []}}
    shipped = {}
    for ki, key in enumerate(keys):
        N = np.random.RandomState(key.rand_seed).randn(3, E.shape[1]); N /= np.linalg.norm(N, axis=1, keepdims=True)
        signs_of = {b: np.array([1.0 if (b >> (2 - j)) & 1 else -1.0 for j in range(3)]) for b in range(8)}
        tally = {K: [0, 0] for K in KS}; tb = {"exact": [0, 0], "random": [0, 0]}; n_pos = 0
        approxN = {K: N @ evecs[:, :K] @ evecs[:, :K].T for K in KS}
        for t in pos:
            for i in range(1, len(t)):
                Xp, cp = t[i - 1]
                pb = semwm.Hyperplanes(np.eye(3)).bin_of(Xp[cp] @ N.T)
                mask = semwm.valid_mask(key, pb)
                X, _ = t[i]
                ex = X @ N.T
                dep = np.array([max(np.min(signs_of[b] * p) for b in mask) for p in ex])
                clr = (dep > 0) & (np.min(np.abs(ex), axis=1) >= semwm.MARGIN)
                n_pos += 1
                for name, j in (("exact", int(np.argmax(dep))), ("random", 0)):
                    tb[name][0] += dep[j] > 0; tb[name][1] += clr[j]
                for K in KS:
                    ap = (mu @ N.T)[None, :] + (X - mu) @ approxN[K].T
                    da = np.array([max(np.min(signs_of[b] * p) for b in mask) for p in ap])
                    j = int(np.argmax(da)); tally[K][0] += dep[j] > 0; tally[K][1] += clr[j]
        if ki == 0:
            shipped = {"n_positions": n_pos, **{f"{nm}_in": tb[nm][0] / n_pos for nm in tb}, **{f"{nm}_clear": tb[nm][1] / n_pos for nm in tb},
                       **{f"K{K}_in": tally[K][0] / n_pos for K in KS}, **{f"K{K}_clear": tally[K][1] / n_pos for K in KS}}
            continue
        for nm in tb:
            bounds[nm]["in"].append(tb[nm][0] / n_pos); bounds[nm]["clear"].append(tb[nm][1] / n_pos)
        for K in KS:
            sel[K]["in"].append(tally[K][0] / n_pos); sel[K]["clear"].append(tally[K][1] / n_pos)
    out["select_random_keys"] = {"n_keys": N_KEYS, **{nm: {m: float(np.mean(v)) for m, v in d.items()} for nm, d in bounds.items()},
                                 **{f"K{K}": {m: float(np.mean(v)) for m, v in sel[K].items()} for K in KS}}
    out["select_shipped_key"] = shipped
    write_json("runs/basis_feasibility.json", out)
    print(f"\nwithin-context variance captured by the top K eigvecs (fit set): " + ", ".join(f"K{k_} {v:.3f}" for k_, v in out["eig_share"].items()))
    print(f"{'K':>5s} {'within R2 mean':>15s} {'R2 p5':>7s} {'sign agree':>11s} {'select: in mask':>16s} {'in mask+clear':>14s}   (random keys, held-out)")
    for K in KS:
        s = out["select_random_keys"][f"K{K}"]
        print(f"{K:5d} {out['K'][K]['within_r2_mean']:15.3f} {out['K'][K]['within_r2_p5']:7.3f} {out['K'][K]['sign_agree']:11.3f} {s['in']:16.3f} {s['clear']:14.3f}")
    for nm in ("exact", "random"):
        s = out["select_random_keys"][nm]; print(f"{nm:>5s} selection: in mask {s['in']:.3f}, in mask+clear {s['clear']:.3f}")
    print("shipped key:", {a: round(b, 3) if isinstance(b, float) else b for a, b in shipped.items()})


if __name__ == "__main__":
    main()
