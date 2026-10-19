"""Pooled-null evaluation of the within-text t-statistic detector for one or more runs (GPU; labelled diagnostic).

For every run: the base texts are scored under EVERY target key of the run's key set (per-token contrasts -> t), giving
a null of n_base x n_keys values (correlated across keys for the same text, so this is still a toy calibration, but
20x larger than the assigned-key null).  A second, stricter null takes the steered texts under every WRONG key
(a text watermarked for key A must not be flagged for key B).  Positives are the steered texts under their assigned
key.  Reports thresholds at 1% and 5% of each null, the TPR at those thresholds and the AUROC.

Usage: python scripts/t_null_pool.py --config configs/pilot.yaml --run runs/abl_poe5c05 [--run ...] [--device cuda]
Writes runs/<name>/eval/t_null_pool.json and prints one summary line per run/arm.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from anywhere without installing the package

from wmworld.config import load_config  # noqa: E402
from wmworld.records import read_json, write_json
from wmworld.scoring import LikelihoodScorer
from wmworld.stages import build_backend, collect_run_texts, load_key_records


def tstat(d: list[float] | None) -> float | None:
    if not d or len(d) < 2:
        return None
    a = np.asarray(d, dtype=np.float64)
    sd = float(a.std(ddof=1))
    return float(a.mean() / (sd / math.sqrt(a.size))) if sd > 0 else None


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p = np.asarray(pos); n = np.asarray(neg)
    gt = (p[:, None] > n[None, :]).mean(); eq = (p[:, None] == n[None, :]).mean()
    return float(gt + 0.5 * eq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-decoys", type=int, default=0, help="also evaluate the decoy-relative t (log p_key - mean of N same-family decoys)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.validate(require_revision=True)
    backend = build_backend(cfg, args.device)
    scorer = LikelihoodScorer(backend, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    for run in args.run:
        t0 = time.time()
        rd = Path(run)
        manifest = read_json(rd / "manifest.json")
        keys = load_key_records(cfg, manifest.get("key_set") or "development")
        specs = {k.key_id: k.to_keyspec() for k in keys}
        gens, _ = collect_run_texts(rd)
        gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("visible_tokens", 0) >= int(cfg["detection"]["min_visible_tokens"])]
        by_arm: dict[str, list[dict]] = {}
        for g in gens:
            by_arm.setdefault(g["arm"], []).append(g)
        # t under every key for every text (cache by text hash)
        t_all: dict[tuple[str, str], float | None] = {}
        texts = {g["text_hash"]: g["text"] for g in gens}
        hashes = sorted(texts)
        lp_key: dict[tuple[str, str], list[float] | None] = {}
        for kid, ks in specs.items():
            for s in range(0, len(hashes), 32):
                chunk = hashes[s:s + 32]
                ds = scorer.per_token_contrasts([texts[h] for h in chunk], ks)
                for h, d in zip(chunk, ds):
                    t_all[(h, kid)] = tstat(d)
                if args.n_decoys > 0:
                    ctx = scorer.context_ids(ks, neutral=False)
                    toks = [scorer.tokenize(texts[h]) for h in chunk]
                    lps = backend.logprob_tokens(ctx, toks)
                    for h, lp in zip(chunk, lps):
                        lp_key[(h, kid)] = lp
        t_spec: dict[tuple[str, str], float | None] = {}
        if args.n_decoys > 0:
            from wmworld.evaluation import load_decoys
            bank = load_decoys(cfg, manifest.get("key_set"))[: args.n_decoys]
            lp_dec: dict[str, list[list[float]]] = {}
            for dk in bank:
                ctx = scorer.context_ids(dk.to_keyspec(), neutral=False)
                for s in range(0, len(hashes), 32):
                    chunk = hashes[s:s + 32]
                    lps = backend.logprob_tokens(ctx, [scorer.tokenize(texts[h]) for h in chunk])
                    for h, lp in zip(chunk, lps):
                        lp_dec.setdefault(h, []).append(lp)
            for (h, kid), lp in lp_key.items():
                dl = lp_dec.get(h)
                if lp and dl:
                    m = np.mean(np.asarray(dl), axis=0)
                    t_spec[(h, kid)] = tstat([a - b for a, b in zip(lp, m)])
            print(f"  decoy-relative t uses {len(bank)} decoys from bank {bank[0].key_set if bank else 'none'}", flush=True)
        out = {"run": str(rd), "n_keys": len(specs), "n_texts": len(hashes), "per_arm": {}, "seconds": None,
               "note": "toy pooled null: every base text under every target key (correlated); wrong-key null: steered texts under every other key"}
        base = by_arm.get("base", [])
        null_base = [t_all[(g["text_hash"], kid)] for g in base for kid in specs if t_all.get((g["text_hash"], kid)) is not None]
        for arm, gs in by_arm.items():
            pos = [t_all[(g["text_hash"], g["assigned_key_id"])] for g in gs if g.get("assigned_key_id") in specs and t_all.get((g["text_hash"], g["assigned_key_id"])) is not None]
            wrong = [t_all[(g["text_hash"], kid)] for g in gs for kid in specs if kid != g.get("assigned_key_id") and t_all.get((g["text_hash"], kid)) is not None]
            rec = {"n_pos": len(pos), "t_pos_mean": float(np.mean(pos)) if pos else None,
                   "null_base": {"n": len(null_base), "mean": float(np.mean(null_base)) if null_base else None, "sd": float(np.std(null_base, ddof=1)) if len(null_base) > 1 else None},
                   "null_wrong_key": {"n": len(wrong), "mean": float(np.mean(wrong)) if wrong else None, "sd": float(np.std(wrong, ddof=1)) if len(wrong) > 1 else None},
                   "auroc_vs_base_null": auroc(pos, null_base), "auroc_vs_wrong_key_null": auroc(pos, wrong)}
            for name, null in (("base", null_base), ("wrong_key", wrong)):
                if not null:
                    continue
                arr = np.sort(np.asarray(null))
                for alpha in (0.01, 0.05):
                    # conservative rank threshold: flag if fewer than alpha*(m+1) - 1 null values are >= s
                    k = int(math.floor(alpha * (len(arr) + 1))) - 1
                    thr = float(arr[len(arr) - 1 - k]) if k >= 0 else float("inf")
                    tpr = float(np.mean([p > thr for p in pos])) if pos else None
                    rec[f"thr_{name}_alpha{alpha}"] = thr
                    rec[f"tpr_{name}_alpha{alpha}"] = tpr
            # FALSE ATTRIBUTION: steered texts tested under a wrong key, flagged at the clean-text threshold
            for alpha in (0.01, 0.05):
                thr = rec.get(f"thr_base_alpha{alpha}")
                rec[f"false_attribution_rate_alpha{alpha}"] = float(np.mean([w > thr for w in wrong])) if (wrong and thr is not None) else None
            rec["t_wrong_key_values"] = [float(x) for x in wrong]      # kept for re-analysis
            rec["t_pos_values"] = [float(x) for x in pos]
            if t_spec:
                sp_pos = [t_spec[(g["text_hash"], g["assigned_key_id"])] for g in gs if t_spec.get((g["text_hash"], g["assigned_key_id"])) is not None]
                sp_wrong = [t_spec[(g["text_hash"], kid)] for g in gs for kid in specs if kid != g.get("assigned_key_id") and t_spec.get((g["text_hash"], kid)) is not None]
                sp_null = [t_spec[(g["text_hash"], kid)] for g in base for kid in specs if t_spec.get((g["text_hash"], kid)) is not None]
                if sp_null and sp_pos:
                    arr = np.sort(np.asarray(sp_null)); k = int(math.floor(0.01 * (len(arr) + 1))) - 1
                    thr = float(arr[len(arr) - 1 - k]) if k >= 0 else float("inf")
                    rec["specific"] = {"n_decoys": args.n_decoys, "t_pos_mean": float(np.mean(sp_pos)), "thr_alpha0.01": thr,
                                       "tpr_alpha0.01": float(np.mean([p > thr for p in sp_pos])),
                                       "false_attribution_rate_alpha0.01": float(np.mean([w > thr for w in sp_wrong])) if sp_wrong else None,
                                       "auroc_vs_base_null": auroc(sp_pos, sp_null), "auroc_vs_wrong_key": auroc(sp_pos, sp_wrong)}
                    sp = rec["specific"]
                    print(f"{rd.name:22s} {arm:12s} DECOY-RELATIVE t: t_pos={sp['t_pos_mean']:5.2f} TPR@1%={sp['tpr_alpha0.01']:.2f} "
                          f"false-attribution@1%={sp['false_attribution_rate_alpha0.01'] if sp['false_attribution_rate_alpha0.01'] is not None else float('nan'):.2f} "
                          f"AUROC={sp['auroc_vs_base_null']:.3f} AUROC(correct vs wrong)={sp['auroc_vs_wrong_key'] or float('nan'):.3f}", flush=True)
            out["per_arm"][arm] = rec
            print(f"{rd.name:22s} {arm:12s} n={len(pos):3d} t_pos={rec['t_pos_mean'] if pos else float('nan'):6.2f} "
                  f"| base null n={len(null_base)} mean={rec['null_base']['mean'] if null_base else float('nan'):6.2f} "
                  f"TPR@1%={rec.get('tpr_base_alpha0.01', float('nan')):.2f} TPR@5%={rec.get('tpr_base_alpha0.05', float('nan')):.2f} AUROC={rec['auroc_vs_base_null'] or float('nan'):.3f} "
                  f"| wrong-key: false-attribution@1%={rec.get('false_attribution_rate_alpha0.01') if rec.get('false_attribution_rate_alpha0.01') is not None else float('nan'):.2f} "
                  f"AUROC(correct vs wrong key)={rec['auroc_vs_wrong_key_null'] or float('nan'):.3f}", flush=True)
        out["seconds"] = time.time() - t0
        (rd / "eval").mkdir(exist_ok=True)
        write_json(rd / "eval" / "t_null_pool.json", out)


if __name__ == "__main__":
    main()
