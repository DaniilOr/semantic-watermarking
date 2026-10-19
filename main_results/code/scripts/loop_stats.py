"""Cleaner repetition measure per arm (text only, no model): word-level loops that ignore markdown/punctuation structure.

For each text: words = lowercase alphabetic tokens; a 'loop' is any 8-word sequence occurring twice or more.
Reports per run/arm: fraction of texts with a loop, mean repeated-8-gram occurrences per 100 words, distinct-word ratio,
and the paired difference vs base on the same prompts.  Usage: python scripts/loop_stats.py runs/a runs/b ...
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[a-z]+")


def stats(text: str, n: int = 8) -> dict:
    w = WORD.findall(text.lower())
    if len(w) < n + 1:
        return {"loop": 0.0, "rep100": 0.0, "distinct": 1.0}
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    rep = sum(v - 1 for v in c.values() if v > 1)
    return {"loop": float(any(v > 1 for v in c.values())), "rep100": 100.0 * rep / len(w), "distinct": len(set(w)) / len(w)}


def main(runs: list[str]) -> None:
    print(f"{'run':24s} {'arm':10s} {'n':>3s} {'loop%':>6s} {'d_loop':>7s} {'rep/100w':>8s} {'d_rep':>7s} {'distinct':>8s}")
    for run in runs:
        gens = [json.loads(l) for l in open(Path(run) / "generations.jsonl") if l.strip()]
        by_arm: dict[str, dict[str, dict]] = {}
        for g in gens:
            if g.get("status", "ok") != "ok":
                continue
            by_arm.setdefault(g["arm"], {})[g["prompt_id"]] = stats(g["text"])
        base = by_arm.get("base", {})
        for arm, d in by_arm.items():
            vals = list(d.values())
            m = lambda k: sum(v[k] for v in vals) / len(vals)
            dl = [d[p]["loop"] - base[p]["loop"] for p in d if p in base]
            dr = [d[p]["rep100"] - base[p]["rep100"] for p in d if p in base]
            print(f"{Path(run).name:24s} {arm:10s} {len(vals):3d} {100 * m('loop'):6.1f} {100 * sum(dl) / max(1, len(dl)):+7.1f} "
                  f"{m('rep100'):8.2f} {sum(dr) / max(1, len(dr)):+7.2f} {m('distinct'):8.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
