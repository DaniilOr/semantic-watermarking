"""Watermark capacity vs semantic uncertainty per sentence position (paper figure; zero generation).

Data: the logged candidate corpora -- 16 candidates per sentence position after the first, from the Llama-3.2-3B on the
training prompts: runs/ss_corpus (plain sampling) and runs/ss_corpus_push (the push, beta 1).  Per position:
  H_bin      entropy (nats) of the empirical distribution of the 16 candidates over the key's 8 bins -- how many
             distinct meaning regions the model's plausible next sentences cover (a key-dependent proxy for semantic
             entropy; 0 = all candidates in one bin, max ln 8 = 2.08)
  p_clear    share of the 16 candidates in the position's mask AND clear of 0.02 (per-draw success)
  any_clear  at least one of the 16 is usable (what bounded selection with cap 16 needs)
Per H_bin bucket and corpus: share of positions, mean per-draw success, share with >= 1 usable candidate among 16 and
among the first 4, share with none among 16, exact-duplicate share.  Writes runs/capacity_entropy.json and the figure
runs/capacity_entropy.png.
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld import semwm  # noqa: E402

PL = semwm.Hyperplanes(np.eye(3))                                         # bin_signs / bin_of only need the plane count
EDGES = [0.0, 1e-9, 0.5, 1.0, 1.5, 2.1]
LABELS = ["0", "(0, 0.5]", "(0.5, 1.0]", "(1.0, 1.5]", "(1.5, 2.08]"]


def positions(run):
    rows = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); rows[r["prompt_id"]] = r
    out = []
    for r in rows.values():
        for u in r["units"]:
            if u["mask"] is None or len(u["cands"]) < 16:
                continue
            P = [np.array(c["proj"]) for c in u["cands"][:16]]
            bins = [PL.bin_of(q) for q in P]
            cnt = np.bincount(bins, minlength=8) / len(bins)
            h = float(-sum(x * math.log(x) for x in cnt if x > 0))
            dep = [semwm.signed_distance(q, tuple(u["mask"]), PL) for q in P]
            clr = [d > 0 and np.min(np.abs(q)) >= semwm.MARGIN for d, q in zip(dep, P)]
            out.append({"h": h, "p_in": float(np.mean([d > 0 for d in dep])), "p_clear": float(np.mean(clr)),
                        "any_clear": any(clr), "any_clear4": any(clr[:4]), "any_in": any(d > 0 for d in dep),
                        "dup": 1 - len({tuple(c["proj"]) for c in u["cands"][:16]}) / 16})
    return out


def table(pos):
    h = np.array([p["h"] for p in pos])
    res = []
    for i, lab in enumerate(LABELS):
        sel = [p for p, x in zip(pos, h) if (x <= EDGES[1] if i == 0 else EDGES[i] < x <= EDGES[i + 1])]
        if not sel:
            res.append({"bucket": lab, "n": 0}); continue
        m = lambda k: float(np.mean([p[k] for p in sel]))
        res.append({"bucket": lab, "n": len(sel), "share": len(sel) / len(pos), "p_in": m("p_in"), "p_clear": m("p_clear"),
                    "any_in16": m("any_in"), "any_clear16": m("any_clear"), "any_clear4": m("any_clear4"),
                    "none_clear16": 1 - m("any_clear"), "dup": m("dup")})
    return res


def main():
    out = {}
    for name, run in (("plain", "runs/ss_corpus"), ("push", "runs/ss_corpus_push")):
        pos = positions(run)
        h = np.array([p["h"] for p in pos]); ac = np.array([p["any_clear"] for p in pos], float)
        rho = float(np.corrcoef(np.argsort(np.argsort(h)), np.argsort(np.argsort(ac)))[0, 1])
        out[name] = {"positions": len(pos), "mean_H": float(h.mean()), "spearman_H_anyclear16": rho,
                     "overall": {k: float(np.mean([p[k] for p in pos])) for k in ("p_in", "p_clear", "any_clear", "any_clear4", "dup")},
                     "buckets": table(pos)}
        print(f"\n== {name}: {len(pos)} positions, mean bin entropy {h.mean():.3f} nats, Spearman(H, >=1 usable of 16) {rho:.3f}")
        print(f"{'H_bin (nats)':>12s} {'share':>6s} {'per-draw usable':>16s} {'>=1 usable /4':>14s} {'>=1 usable /16':>15s} {'none /16':>9s} {'dup':>6s}")
        for b in out[name]["buckets"]:
            if b["n"]:
                print(f"{b['bucket']:>12s} {b['share']:6.3f} {b['p_clear']:16.3f} {b['any_clear4']:14.3f} {b['any_clear16']:15.3f} {b['none_clear16']:9.3f} {b['dup']:6.3f}")
    Path("runs/capacity_entropy.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    col = {"plain": "#2a78d6", "push": "#eb6834"}                          # validated categorical slots 1-2
    lab = {"plain": "plain sampling", "push": "with the push"}
    x = np.arange(len(LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=surf)
    panels = [("share", "share of sentence positions"), ("p_clear", "per-draw usable (in mask, clear of 0.02)"),
              ("any_clear16", ">= 1 usable among 16 candidates")]
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(surf)
        for name in ("plain", "push"):
            ys = [b.get(key, np.nan) if b["n"] else np.nan for b in out[name]["buckets"]]
            if key == "share":
                off = -0.18 if name == "plain" else 0.18
                ax.bar(x + off, ys, width=0.34, color=col[name], label=lab[name], edgecolor=surf, linewidth=2)
            else:
                ax.plot(x, ys, color=col[name], lw=2, marker="o", ms=8, markeredgecolor=surf, markeredgewidth=2, label=lab[name])
                ax.annotate(f"{ys[-1]:.2f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), va="center", color=ink2, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8, color=ink2)
        ax.set_xlabel("bin entropy of the position's 16 candidates (nats)", fontsize=9, color=ink2)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_ylim(0, 1.0 if key == "any_clear16" else None)
        ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle("Watermark capacity follows the model's semantic uncertainty (Llama-3.2-3B, training prompts)", fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("runs/capacity_entropy.png", dpi=200, facecolor=surf)
    fig.savefig("runs/capacity_entropy.pdf", facecolor=surf)
    print("wrote runs/capacity_entropy.{json,png,pdf}")


if __name__ == "__main__":
    main()
