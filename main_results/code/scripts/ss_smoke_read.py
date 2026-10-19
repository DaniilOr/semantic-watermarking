"""Print every smoke text with the numbers that gate an arm: loops, length, boundaries, steering mechanics, depth."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402

run = Path(sys.argv[1]); out_md = run / "samples.md"
emb = semwm.Embedder(sys.argv[2] if len(sys.argv) > 2 else
                     "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282", device="cpu")
key = semwm.Key.shipped(); pl = semwm.Hyperplanes.for_key(key, emb.dim); tb = semwm.fixed_bin(key)
by_arm = defaultdict(list)
for f in sorted(run.glob("generations.*.jsonl")):
    for ln in open(f):
        g = json.loads(ln); by_arm[g["arm"]].append(g)


def max_repeat(text: str, n: int = 4) -> int:
    w = text.split()
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values()) if c else 0


md = [f"# Smoke samples: {run}\n"]
print(f"{'arm':16s} {'n':>2s} {'tok':>5s} {'words/sent':>10s} {'maxrep4':>7s} {'t_seg':>7s} {'|b|':>5s} {'Pb0':>5s} {'Pb1':>5s} {'rep':>4s} {'align':>7s} {'s/text':>6s}")
for arm, gs in sorted(by_arm.items()):
    toks = [g["visible_tokens"] for g in gs]
    wps = [len(g["text"].split()) / max(1, len(semwm.split_sentences(g["text"]))) for g in gs]
    reps = [max_repeat(g["text"]) for g in gs]
    tseg = [semwm.seg_stat(pl.projections(emb.encode(semwm.split_segments(g["text"], 16))), tb, pl) or 0.0 for g in gs]
    c = lambda k: np.mean([g["cost"].get(k) or 0 for g in gs])
    al = f"{sum(g['cost'].get('align_ok', 0) for g in gs)}/{sum(g['cost'].get('align_units', 0) for g in gs)}" if "align_units" in gs[0]["cost"] else "-"
    print(f"{arm:16s} {len(gs):2d} {np.median(toks):5.0f} {np.median(wps):10.1f} {max(reps):7d} {np.mean(tseg):7.3f} {c('steer_bias_abs_mean'):5.2f} "
          f"{c('steer_p_boundary_before'):5.3f} {c('steer_p_boundary_after'):5.3f} {int(sum(g['cost'].get('steer_rep_guard_hits', 0) for g in gs)):4d} {al:>7s} {c('wall_seconds'):6.0f}")
    md.append(f"\n## {arm}\n")
    for g in sorted(gs, key=lambda g: g["prompt_id"]):
        md.append(f"\n**{g['prompt_id']}** ({g['visible_tokens']} tokens, max 4-gram repeat {max_repeat(g['text'])})\n\n{g['text']}\n")
out_md.write_text("\n".join(md))
print(f"\nwrote {out_md}")
