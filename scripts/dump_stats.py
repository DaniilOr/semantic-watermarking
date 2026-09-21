import argparse
import time
from pathlib import Path

import _path
from wmark import Reference, position_stats, write_dump
from wmark.io import load_generations
from wmark.keys import load_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--extra-keys", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--token-budget", type=int, default=3072)
    ap.add_argument("--name", default="posstats")
    ap.add_argument("--ref-cache-dir", default=None)
    a = ap.parse_args()
    keys = load_keys(a.keys) + (load_keys(a.extra_keys) if a.extra_keys else [])
    reference = Reference(cache_dir=a.ref_cache_dir)
    for run in a.run:
        t0 = time.time()
        gens = [g for g in load_generations(run) if int(g.get("n_tokens") or 0) >= a.min_tokens]
        if a.arms:
            gens = [g for g in gens if g["arm"] in set(a.arms.split(","))]
        stats = []
        for s in range(0, len(gens), 64):
            stats += position_stats(reference, keys, [g["text"] for g in gens[s:s + 64]], a.token_budget)
            print(f"[{Path(run).name}] {min(s + 64, len(gens))}/{len(gens)} ({time.time() - t0:.0f}s)", flush=True)
        dest = Path(run) / f"{a.name}.npz"
        write_dump(dest, keys, gens, stats)
        print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
