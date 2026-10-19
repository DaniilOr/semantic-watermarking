"""Freeze a per-key clean null for the cross-model detector from the dedicated null partition.

Why this exists: `cross_detect.py` estimates its 1% threshold from the *current run's* own ``base`` arm -- 60
texts scored under every key.  Those 60 texts give ~720 values, but they are 60 independent texts, so the 1%
tail rests on roughly six independent observations and the threshold is extremely noisy.  Two runs of the same
configuration produced thresholds of 2.60 and 3.85, which moved a fixed signal (T = 3.2) from "detected 87% of
the time" to "detected 5% of the time".  That is threshold noise being read as a method difference.

This scores the ``null_calibration`` partition (1,200 independent clean texts) once per key and saves the
resulting null, so every run is compared against the same frozen, adequately-sized reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import collect_run_texts, load_key_records  # noqa: E402


def tstat(d):
    if not d or len(d) < 2:
        return None
    a = np.asarray(d, dtype=np.float64)
    sd = float(a.std(ddof=1))
    return float(a.mean() / (sd / math.sqrt(a.size))) if sd > 0 else None


def upper_tail_threshold(values: list[float], alpha: float) -> float:
    """Conservative rank threshold: the smallest t with #{v >= t} <= floor(alpha*(m+1)) - 1."""
    a = np.sort(np.asarray(values, dtype=np.float64))
    k = int(math.floor(alpha * (len(a) + 1))) - 1
    return float(a[len(a) - 1 - k]) if k >= 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--null-run", default="runs/null_calibration")
    ap.add_argument("--key-set", default="randomwords")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of clean texts (0 = all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.validate(require_revision=True)
    ref = build_reference(cfg)
    scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
    keys = load_key_records(cfg, args.key_set)
    specs = {k.key_id: k.to_keyspec() for k in keys}

    gens, _ = collect_run_texts(Path(args.null_run))
    gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("visible_tokens", 0) >= args.min_tokens]
    if args.limit:
        gens = gens[: args.limit]
    texts = {g["text_hash"]: g["text"] for g in gens}
    lengths = {g["text_hash"]: int(g["visible_tokens"]) for g in gens}
    hashes = sorted(texts)
    print(f"clean texts: {len(hashes)} from {args.null_run}; keys: {len(specs)}", flush=True)

    t0 = time.time()
    per_key: dict[str, list[float]] = {}
    per_key_len: dict[str, list[int]] = {}
    for kid, ks in specs.items():
        vals, lens = [], []
        for s in range(0, len(hashes), 32):
            chunk = hashes[s:s + 32]
            for h, d in zip(chunk, scorer.per_token_contrasts([texts[h] for h in chunk], ks)):
                t = tstat(d)
                if t is not None:
                    vals.append(t); lens.append(lengths[h])
        per_key[kid] = vals
        per_key_len[kid] = lens
        print(f"  {kid}: n={len(vals)} mean={np.mean(vals):+.3f} sd={np.std(vals, ddof=1):.3f} "
              f"thr={upper_tail_threshold(vals, args.alpha):.2f} ({time.time() - t0:.0f}s)", flush=True)

    pooled = [v for vals in per_key.values() for v in vals]
    out = {
        "reference": ref.info,
        "null_run": args.null_run,
        "key_set": args.key_set,
        "alpha": args.alpha,
        "min_tokens": args.min_tokens,
        "n_texts": len(hashes),
        "per_key": {k: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                        "threshold": upper_tail_threshold(v, args.alpha),
                        "values": [round(x, 4) for x in v],
                        "lengths": per_key_len[k]} for k, v in per_key.items()},
        "pooled": {"n": len(pooled), "mean": float(np.mean(pooled)), "sd": float(np.std(pooled, ddof=1)),
                   "threshold": upper_tail_threshold(pooled, args.alpha)},
    }
    dest = Path(args.out or f"assets/cross_null_{args.key_set}.json")
    write_json(dest, out)
    thr = [out["per_key"][k]["threshold"] for k in per_key]
    print(f"\nwrote {dest}: {len(hashes)} texts x {len(specs)} keys")
    print(f"pooled threshold T>{out['pooled']['threshold']:.2f} | per-key thresholds "
          f"min={min(thr):.2f} median={float(np.median(thr)):.2f} max={max(thr):.2f}")


if __name__ == "__main__":
    main()
