"""Horizontal 100%-stacked bars of the pairwise judge's verdicts (Gemma-4-31B, both presentation orders): watermarked
text better / tie / plain text better, per task (rows) and chat model (columns), for the second plain sample (the
judge's no-difference control), SemStamp and Push + Margin.  Read-only; writes a PNG.
Usage: python scripts/judge_bars.py OUT.png
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIN, TIE, LOSS = "#2a78d6", "#d4d3ce", "#e34948"            # diverging poles (validated) + neutral midpoint
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
MODELS = [("llama3b", "Llama-3.2-3B"), ("llama1b", "Llama-3.2-1B"), ("qwen08b", "Qwen3.5-0.8B")]
ARMS = [("clean_seed1", "2nd plain sample"), ("semstamp", "SemStamp"), ("chain_reference", "Push + Margin")]
TASKS = [("it_writingprompts", "WritingPrompts"), ("it_dolly", "Dolly"), ("it_ifeval", "IFEval"), ("it_gsm8k", "GSM8K"),
         ("it_translation", "Translation de→en"), ("ob", "Continuation: our benchmark"), ("sp", "Continuation: SemStamp's datasets")]
RUN = {("ob", "llama3b"): "ob_bench", ("ob", "llama1b"): "ob_bench_llama1b", ("ob", "qwen08b"): "ob_bench_qwen08b_nl",
       ("sp", "llama3b"): "sp_bench_llama3b", ("sp", "llama1b"): "sp_bench_llama1b", ("sp", "qwen08b"): "sp_bench_qwen08b_nl"}
_cache = {}


def rows(run):
    if run not in _cache:
        f = Path(f"runs/{run}/eval/api_pref_google_gemma-4-31b-it.jsonl")
        _cache[run] = [json.loads(l) for l in open(f, encoding="utf-8")] if f.exists() else []
    return _cache[run]


def shares(task, model, arm):
    if task in ("ob", "sp"):
        rs = [r for r in rows(RUN[(task, model)]) if r["arm"] == arm and r.get("domain") != "sp_mbpp"]
    else:
        rs = [r for r in rows(f"it_bench_{model}") if r["arm"] == arm and r.get("domain") == task]
    c = Counter("tie" if r["judgment"]["better"] == "tie" else ("win" if r["judgment"]["better"] == r["wm_is"] else "loss") for r in rs)
    n = sum(c.values())
    return (None, 0) if not n else ([100 * c[k] / n for k in ("win", "tie", "loss")], n)


def main():
    out = sys.argv[1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "text.color": INK})
    fig, axes = plt.subplots(len(TASKS), len(MODELS), figsize=(13, 1.25 * len(TASKS) + 1.4), sharex=True, facecolor=SURF)
    for i, (task, tname) in enumerate(TASKS):
        for j, (m, mname) in enumerate(MODELS):
            ax = axes[i, j]; ax.set_facecolor(SURF)
            for k, (arm, aname) in enumerate(ARMS):
                y = len(ARMS) - 1 - k
                s, n = shares(task, m, arm)
                if s is None:
                    ax.text(1, y, "not judged", va="center", fontsize=8, color=INK2)
                    continue
                left = 0
                for v, col in zip(s, (WIN, TIE, LOSS)):
                    ax.barh(y, v, left=left, height=0.62, color=col, edgecolor=SURF, linewidth=2)
                    if v >= 9:
                        ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                                color="#ffffff" if col != TIE else INK)
                    left += v
            ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(ARMS) - 0.4)
            ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([a for _, a in ARMS][::-1] if j == 0 else [""] * len(ARMS), color=INK2)
            ax.axvline(50, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=INK2)
            if i == 0:
                ax.set_title(mname, fontsize=10, fontweight="bold", color=INK)
            if j == 0:
                ax.annotate(tname, xy=(-0.42, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=9.5, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("% of judge verdicts", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    fig.legend(handles, ["watermarked text better", "tie", "plain text better"], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 1.0), fontsize=9.5)
    fig.suptitle("Pairwise judge (Gemma-4-31B, both orders): watermarked vs plain answer to the same prompt", y=1.035, fontsize=11)
    fig.text(0.55, -0.01, "Numbers: % of verdicts (shown when ≥ 9%). The 2nd plain sample is the no-difference control: two plain answers judged "
             "the same way. Dashed line = 50%.", ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.93, bottom=0.07, hspace=0.35, wspace=0.06)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
