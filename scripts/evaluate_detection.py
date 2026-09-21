import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import _path
from wmark import Dump
from wmark.io import load_generations, write_json
from wmark.keys import load_keys
from wmark.stats import VARIANTS, auroc, upper_tail_threshold, wilson


def null_table(null, keys, variant, alpha, lo=0, hi=None, min_len=0, min_null=100):
    out = {}
    for k in keys:
        v = [null.stat(i, k, variant, lo, hi) for i in range(len(null)) if null.n_tokens(i) >= min_len]
        v = [x for x in v if x is not None]
        if len(v) >= min_null:
            out[k] = {"thr": upper_tail_threshold(v, alpha), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                      "n": len(v), "values": v}
    return out


def evaluate(run, null, keys, variant, alpha, clean_arms, lo=0, hi=None, min_len=0, min_null=100):
    nt = null_table(null, keys, variant, alpha, lo, hi, min_len, min_null)
    std = lambda s, k: (s - nt[k]["mean"]) / nt[k]["sd"]
    null_std = [std(x, k) for k in nt for x in nt[k]["values"]]
    by_arm = defaultdict(list)
    for i in range(len(run)):
        if run.n_tokens(i) >= min_len:
            by_arm[run.arm[i]].append(i)
    res = {"thresholds": {k: nt[k]["thr"] for k in nt}}
    for arm, idx in sorted(by_arm.items()):
        pos, wrong, clean, flagged = [], [], [], set()
        for i in idx:
            for k in nt:
                s = run.stat(i, k, variant, lo, hi)
                if s is None:
                    continue
                if arm in clean_arms:
                    clean.append((s, k))
                    if s > nt[k]["thr"]:
                        flagged.add(i)
                elif k == run.key_id[i]:
                    pos.append((s, k))
                else:
                    wrong.append((s, k))
        rec = {"n_texts": len(idx)}
        if pos:
            hits = sum(s > nt[k]["thr"] for s, k in pos)
            ci = wilson(hits, len(pos))
            rec.update({"n": len(pos), "tpr": hits / len(pos), "tpr_ci95": list(ci),
                        "z_mean": float(np.mean([std(s, k) for s, k in pos])),
                        "auroc": auroc([std(s, k) for s, k in pos], null_std),
                        "wrong_key_rate": (sum(s > nt[k]["thr"] for s, k in wrong) / len(wrong)) if wrong else None})
        if clean:
            fp = sum(s > nt[k]["thr"] for s, k in clean)
            rec.update({"n_pairs": len(clean), "fp": fp, "fpr": fp / len(clean), "texts_flagged_any_key": len(flagged)})
        res[arm] = rec
    return res


def rotation_recovery(run, run_dir, variant, min_span=24):
    gens = {g["id"]: g for g in load_generations(run_dir)}
    out = defaultdict(lambda: {"spans": 0, "correct": 0, "texts": 0, "n_keys": 0})
    for i in range(len(run)):
        g = gens.get(run.gen_id[i])
        if not g or len(g.get("key_ids") or []) < 2 or any(k not in run.kidx for k in g["key_ids"]):
            continue
        n_ref, n_gen = run.n_tokens(i), max(1, int(g["n_tokens"]))
        starts = [int(s) for s in g["span_starts"]] + [n_gen]
        rec = out[run.arm[i]]
        rec["texts"] += 1
        rec["n_keys"] = len(g["key_ids"])
        for j, true_idx in enumerate(g["key_order"]):
            lo, hi = int(round(starts[j] * n_ref / n_gen)), int(round(starts[j + 1] * n_ref / n_gen))
            if hi - lo < min_span:
                continue
            sc = [run.stat(i, k, variant, lo, hi) for k in g["key_ids"]]
            if any(s is None for s in sc):
                continue
            rec["spans"] += 1
            rec["correct"] += int(int(np.argmax(sc)) == int(true_idx))
    return {a: {**r, "accuracy": r["correct"] / r["spans"] if r["spans"] else None,
                "chance": 1.0 / r["n_keys"] if r["n_keys"] else None} for a, r in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--name", default="posstats")
    ap.add_argument("--variants", default="clip_gate,plain")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-null", type=int, default=100)
    ap.add_argument("--clean-arms", default="base")
    ap.add_argument("--prefixes", default="")
    ap.add_argument("--rotation", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    null = Dump(a.null)
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    clean_arms = set(a.clean_arms.split(","))
    main_keys = {k.key_id for k in load_keys(a.keys)} if a.keys else set()
    report = {"null": a.null, "null_texts": len(null), "alpha": a.alpha, "runs": {}}
    for rd in a.run:
        rd = Path(rd)
        run = Dump(rd / f"{a.name}.npz")
        keys = [k for k in run.key_ids if k in null.kidx and (not main_keys or k in main_keys)]
        rr = {"variants": {v: evaluate(run, null, keys, v, a.alpha, clean_arms, min_null=a.min_null) for v in variants}}
        for v in variants:
            for arm, r in rr["variants"][v].items():
                if "tpr" in r:
                    print(f"{rd.name} {v:10s} {arm:14s} n={r['n']:4d} TPR={r['tpr']:.3f} [{r['tpr_ci95'][0]:.3f},{r['tpr_ci95'][1]:.3f}] "
                          f"AUROC={r['auroc']:.3f} wrong-key={r['wrong_key_rate']}")
                if "fpr" in r:
                    print(f"{rd.name} {v:10s} {arm:14s} FPR={r['fpr']:.4f} ({r['fp']}/{r['n_pairs']}) texts flagged={r['texts_flagged_any_key']}/{r['n_texts']}")
        if a.prefixes:
            rr["by_prefix"] = {}
            for length in [int(x) for x in a.prefixes.split(",") if x.strip()]:
                rr["by_prefix"][length] = {v: evaluate(run, null, keys, v, a.alpha, clean_arms, 0, length, length, a.min_null) for v in variants}
        if a.rotation:
            rr["rotation"] = {v: rotation_recovery(run, rd, v) for v in variants}
        report["runs"][rd.name] = rr
    dest = Path(a.out or "detection.json")
    write_json(dest, report)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
