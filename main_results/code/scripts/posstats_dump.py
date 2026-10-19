"""Dump per-position detector statistics for a run's texts under every key (one GPU pass; analysis is offline).

For each text: the neutral world's entropy per position.  For each text and key: the realized contrast, and the
mean/variance the contrast would have under the neutral world's own next-token distribution.  With these saved,
every detector variant, every length prefix and every window scan is a cheap array computation
(scripts/detector_variants.py) instead of another model pass.

Usage: python scripts/posstats_dump.py --config configs/pilot.yaml --run runs/<name> --key-set randomwords
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.posstats import write_dump  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import collect_run_texts, load_key_records, steering_decoys  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--key-set", default="randomwords")
    ap.add_argument("--extra", default="", help="'steering_decoys': also score the rotation/decoy phrase pool as pseudo-keys")
    ap.add_argument("--arms", default="", help="comma list; default all arms")
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=3072)
    ap.add_argument("--window", type=int, default=0, help="bounded reference context: last W..W+S-1 tokens (0 = whole text)")
    ap.add_argument("--stride", type=int, default=0, help="block size S of the bounded view (0 = W/2)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.validate(require_revision=True)
    ref = build_reference(cfg)
    scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
    keys = load_key_records(cfg, args.key_set)
    specs = [(k.key_id, k.to_keyspec()) for k in keys]
    if args.extra == "steering_decoys":
        import dataclasses
        base = specs[0][1]
        for dct in steering_decoys(cfg).get("random_words", []):
            specs.append((f"phrase:{dct['prefix']}", dataclasses.replace(base, key_id=dct["key_id"], prefix=dct["prefix"], meta={})))
    neutral = {tuple(scorer.context_ids(ks, neutral=True)) for _, ks in specs}
    assert len(neutral) == 1, "keys do not share one neutral context; dump them per key set"
    ctx0 = list(next(iter(neutral)))
    ctxk = [scorer.context_ids(ks, neutral=False) for _, ks in specs]
    print(f"reference: {ref.info['model_id']} | keys: {len(specs)} | neutral ctx {len(ctx0)} tokens", flush=True)

    for run in args.run:
        t0 = time.time()
        rd = Path(run)
        gens, _ = collect_run_texts(rd)
        gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("visible_tokens", 0) >= args.min_tokens]
        if args.arms:
            keep = set(args.arms.split(","))
            gens = [g for g in gens if g["arm"] in keep]
        if args.limit:
            gens = gens[: args.limit]
        toks = [scorer.tokenize(g["text"]) for g in gens]
        room = ref.max_input_tokens - max(len(c) for c in [ctx0] + ctxk)
        ok = [i for i, t in enumerate(toks) if 2 <= len(t) <= room]
        print(f"[{rd.name}] {len(ok)}/{len(gens)} texts eligible", flush=True)
        stats: list[dict | None] = [None] * len(gens)
        for s in range(0, len(ok), 64):
            chunk = ok[s:s + 64]
            res = (ref.position_stats_windowed(ctx0, ctxk, [toks[i] for i in chunk], args.window, args.stride) if args.window
                   else ref.position_stats(ctx0, ctxk, [toks[i] for i in chunk], token_budget=args.token_budget))
            for i, ps in zip(chunk, res):
                stats[i] = ps
            print(f"  [{rd.name}] {min(s + 64, len(ok))}/{len(ok)} ({time.time() - t0:.0f}s)", flush=True)
        use = [i for i in ok if stats[i] is not None]
        dest = rd / "eval" / (f"posstats_{args.key_set}_w{args.window}.npz" if args.window else f"posstats_{args.key_set}.npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_dump(dest, [k for k, _ in specs], [ks.prefix for _, ks in specs], [gens[i] for i in use],
                   [stats[i] for i in use], reference=ref.info["model_id"])
        n_tok = sum(stats[i]["h0"].shape[0] for i in use)
        print(f"[{rd.name}] wrote {dest} ({len(use)} texts, {n_tok} tokens, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
