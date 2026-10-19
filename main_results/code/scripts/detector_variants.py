"""Offline detector analysis over per-position dumps (CPU only; run it through the CPU queue).

Compares detector variants against ONE frozen clean null (per-key 1% thresholds from the null partition):
true-positive rate with a Wilson interval, false-positive rate on an INDEPENDENT clean cohort (the run's own
base arm), false attribution under wrong keys, key-standardised AUROC, length-matched prefix views, and, for
key-rotation arms, how often the key of each span is recovered.

Usage: python scripts/detector_variants.py --null runs/null_calibration/eval/posstats_randomwords.npz \
           --run runs/cross_llama --run runs/spans [--prefixes 128,256,512] [--config configs/pilot.yaml --rotation]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.posstats import VARIANTS, Dump, auroc, upper_tail_threshold, wilson  # noqa: E402

ALPHA = 0.01


def null_table(null: Dump, keys: list[str], variant: str, lo: int = 0, hi: int | None = None, min_len: int = 0):
    """Per key: clean-null values -> (threshold, mean, sd, n)."""
    out = {}
    for k in keys:
        v = [null.stat(variant, i, k, lo, hi) for i in range(len(null)) if null.n_tokens(i) >= min_len]
        v = [x for x in v if x is not None]
        if len(v) >= 100:
            out[k] = {"thr": upper_tail_threshold(v, ALPHA), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                      "n": len(v), "values": v}
    return out


def evaluate(run: Dump, null: Dump, keys: list[str], variant: str, lo: int = 0, hi: int | None = None, min_len: int = 0) -> dict:
    nt = null_table(null, keys, variant, lo, hi, min_len)
    std = lambda s, k: (s - nt[k]["mean"]) / nt[k]["sd"]
    null_std = [std(x, k) for k in nt for x in nt[k]["values"]]
    by_arm = defaultdict(list)
    for i in range(len(run)):
        if run.n_tokens(i) >= min_len:
            by_arm[run.arm[i]].append(i)
    res = {}
    for arm, idx in sorted(by_arm.items()):
        marked = arm != "base"
        pos, wrong, clean = [], [], []
        for i in idx:
            for k in nt:
                s = run.stat(variant, i, k, lo, hi)
                if s is None:
                    continue
                if not marked:
                    clean.append((s, k))
                elif k == run.assigned[i]:
                    pos.append((s, k))
                else:
                    wrong.append((s, k))
        rec = {"n_texts": len(idx)}
        if marked and pos:
            hits = sum(s > nt[k]["thr"] for s, k in pos)
            lo_, hi_ = wilson(hits, len(pos))
            rec.update({"n": len(pos), "stat_mean": float(np.mean([s for s, _ in pos])),
                        "z_mean": float(np.mean([std(s, k) for s, k in pos])),
                        "tpr": hits / len(pos), "tpr_ci": [lo_, hi_],
                        "auroc": auroc([std(s, k) for s, k in pos], null_std),
                        "false_attribution": (sum(s > nt[k]["thr"] for s, k in wrong) / len(wrong)) if wrong else None,
                        "wrong_z_mean": float(np.mean([std(s, k) for s, k in wrong])) if wrong else None,
                        "pos_z": [round(float(std(s, k)), 3) for s, k in pos],          # per-text, for the figures
                        "thr_z_mean": float(np.mean([std(nt[k]["thr"], k) for k in nt]))})
        if not marked and clean:
            fp = sum(s > nt[k]["thr"] for s, k in clean)
            rec.update({"n_pairs": len(clean), "fpr": fp / len(clean), "fp": fp,
                        "texts_flagged_any_key": len({i for i in idx for k in nt
                                                      if (run.stat(variant, i, k, lo, hi) or -1e9) > nt[k]["thr"]})})
        res[arm] = rec
    hist, edges = np.histogram(np.clip(null_std, -6, 12), bins=90, range=(-6, 12))
    res["_null"] = {"n": len(null_std), "hist": hist.tolist(), "edges": [round(float(e), 3) for e in edges],
                    "n_texts": int(np.mean([nt[k]["n"] for k in nt])) if nt else 0}
    return res


def rotation_recovery(run: Dump, run_dir: Path, cfg_path: str, variant: str) -> dict:
    """For rotation arms: per generated span, is the span's own key the best-scoring of the rotation's keys?"""
    from wmworld.config import load_config
    from wmworld.records import read_json
    from wmworld.stages import keyspec_with_decoys, load_key_records
    cfg = load_config(cfg_path)
    manifest = read_json(run_dir / "manifest.json")
    recs = {k.key_id: k for k in load_key_records(cfg, manifest.get("key_set") or "randomwords")}
    traj = {}
    for f in run_dir.rglob("trajectories*.jsonl"):
        for ln in open(f):
            r = json.loads(ln)
            c = r.get("cost") or {}
            if c.get("n_rotation_keys"):
                traj[str(r["output_id"])] = c
    out = defaultdict(lambda: {"spans": 0, "correct": 0, "texts": 0})
    for i in range(len(run)):
        c = traj.get(run.gen_id[i])
        if not c or run.assigned[i] not in recs:
            continue
        ks = keyspec_with_decoys(cfg, recs[run.assigned[i]])
        m = int(c["n_rotation_keys"])
        phrases = ([ks.prefix] + [p for p in (ks.meta.get("rotation_keys") or []) if p != ks.prefix])[:m]
        kids = [run.assigned[i]] + [f"phrase:{p}" for p in phrases[1:]]     # partners are dumped as pseudo-keys
        if any(k not in run.kidx for k in kids):
            continue
        n_ref, n_gen = run.n_tokens(i), max(1, int(run.n_gen_tokens[i]))
        starts = [int(s) for s in c["span_starts"]] + [n_gen]
        rec = out[run.arm[i]]
        rec["texts"] += 1
        for j, true_idx in enumerate(c["key_order"]):
            a, b = int(round(starts[j] * n_ref / n_gen)), int(round(starts[j + 1] * n_ref / n_gen))
            if b - a < 24:
                continue
            sc = [run.stat(variant, i, k, a, b) for k in kids]
            if any(s is None for s in sc):
                continue
            rec["spans"] += 1
            rec["correct"] += int(int(np.argmax(sc)) == int(true_idx))
    return {a: {**r, "accuracy": (r["correct"] / r["spans"]) if r["spans"] else None, "chance": 0.25} for a, r in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", required=True)
    ap.add_argument("--run", action="append", required=True, help="run directory (uses eval/posstats_<keyset>.npz)")
    ap.add_argument("--key-set", default="randomwords")
    ap.add_argument("--dump-suffix", default="", help="e.g. _w64 for dumps made under a bounded reference context")
    ap.add_argument("--variants", default="", help="comma list; default all")
    ap.add_argument("--prefixes", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--rotation", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    null = Dump(args.null)
    variants = [v for v in (args.variants.split(",") if args.variants else VARIANTS) if v in VARIANTS]
    report = {"null": args.null, "null_texts": len(null), "alpha": ALPHA, "runs": {}}
    for rd in args.run:
        rd = Path(rd)
        path = rd / "eval" / f"posstats_{args.key_set}{args.dump_suffix}.npz"
        run = Dump(path)
        run.path = str(path)
        keys = [k for k in run.key_ids if k in null.kidx and not k.startswith("phrase:")]
        print(f"\n=== {rd.name}: {len(run)} texts, {len(keys)} keys; frozen null {len(null)} texts ===")
        print(f"{'variant':18s} {'arm':12s} {'n':>4s} {'z':>6s} {'TPR@1%':>7s} {'95% CI':>13s} {'AUROC':>6s} {'wrongkey':>8s} | base FPR (pairs)")
        rr = {"variants": {}}
        for v in variants:
            res = evaluate(run, null, keys, v)
            rr["variants"][v] = res
            base = res.get("base", {})
            fpr = f"{base.get('fpr', float('nan')):.4f} ({base.get('fp', 0)}/{base.get('n_pairs', 0)})" if base else "-"
            for arm, r in res.items():
                if arm in ("base", "_null") or "tpr" not in r:
                    continue
                print(f"{v:18s} {arm:12s} {r['n']:4d} {r['z_mean']:6.2f} {r['tpr']:7.2f} "
                      f"[{r['tpr_ci'][0]:.2f},{r['tpr_ci'][1]:.2f}] {r['auroc']:6.3f} "
                      f"{(r['false_attribution'] if r['false_attribution'] is not None else float('nan')):8.4f} | {fpr}")
        if args.prefixes:
            rr["by_prefix"] = {}
            for L in [int(x) for x in args.prefixes.split(",") if x.strip()]:
                rr["by_prefix"][L] = {v: evaluate(run, null, keys, v, 0, L, min_len=L) for v in variants}
                print(f"--- first {L} reference tokens, length-matched null ---")
                for v in variants:
                    for arm, r in rr["by_prefix"][L][v].items():
                        if "tpr" in r:
                            print(f"{v:18s} {arm:12s} n={r['n']:4d} z={r['z_mean']:6.2f} TPR={r['tpr']:.2f} AUROC={r['auroc']:.3f}")
        if args.rotation and args.config:
            rr["rotation"] = {v: rotation_recovery(run, rd, args.config, v) for v in variants[:3]}
            for v, r in rr["rotation"].items():
                for arm, x in r.items():
                    print(f"ROTATION {v:14s} {arm:12s} spans={x['spans']:4d} key recovered={x['accuracy'] if x['accuracy'] is not None else float('nan'):.2f} (chance {x['chance']})")
        report["runs"][rd.name] = rr
    dest = Path(args.out or "runs/detector_variants.json")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k != "values"}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(clean(report), open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
