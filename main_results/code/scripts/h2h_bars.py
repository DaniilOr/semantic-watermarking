"""Horizontal 100%-stacked bars, Push + Margin vs SemStamp on the same prompt: push better / tie / SemStamp better, per
dataset and model, with the win rate (push better + 0.5 tie) and its bootstrap 95% CI.  Read-only: plots
runs/h2h_semstamp.json (scripts/h2h_semstamp.py).

  pointwise   per prompt, which text got the higher 1-5 quality score (equal = tie)
  via_plain   per prompt, which text did better against the same plain answer (pairwise judge, both orders)

Usage: python scripts/h2h_bars.py {pointwise|via_plain} OUT.png
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIN, TIE, LOSS = "#2a78d6", "#d4d3ce", "#e34948"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BASE_CHAT = ["Llama-3B base", "Llama-1B base", "Qwen base", "Llama-3B chat", "Llama-1B chat", "Qwen chat"]
CHAT = ["Llama-3B chat", "Llama-1B chat", "Qwen chat"]
LEFT = [("sp_realnews", "RealNews (SemStamp's)"), ("sp_booksum", "BookSum (SemStamp's)"), ("sp_tifu", "Reddit-TIFU (SemStamp's)"),
        ("ob_arxiv", "ArXiv (ours)"), ("ob_booksum", "BookSum (ours)"), ("ob_wikipedia", "Wikipedia (ours)")]
RIGHT = [("it_writingprompts", "WritingPrompts"), ("it_dolly", "Dolly"), ("it_ifeval", "IFEval"), ("it_gsm8k", "GSM8K (math)"),
         ("it_translation", "Translation de→en"), ("sp_mbpp", "MBPP (code)")]
TITLE = {"pointwise": "Push + Margin vs SemStamp, same prompt: which text got the higher 1–5 quality score (Gemma-4-31B, pointwise)",
         "via_plain": "Push + Margin vs SemStamp, same prompt: which text did better against the same plain answer (Gemma-4-31B pairwise, both orders)"}
NOTE = {"pointwise": "Tie = both texts got the same score.",
        "via_plain": "Per prompt each text scores 1 / 0.5 / 0 per order against the plain answer (mean of 2 orders); tie = equal scores."}


def panel(ax, cells, models, tname, src):
    ax.set_facecolor(SURF)
    for k, m in enumerate(models):
        y = 5 - k
        c = cells.get(m)
        if not c:
            ax.text(1, y, "no pointwise scores" if src == "pointwise" and tname.startswith("Translation") else "not run",
                    va="center", fontsize=8, color=INK2)
            continue
        left = 0
        for v, col in zip((100 * c["p_push_higher"], 100 * c["p_equal"], 100 * c["p_ss_higher"]), (WIN, TIE, LOSS)):
            ax.barh(y, v, left=left, height=0.66, color=col, edgecolor=SURF, linewidth=1.5)
            if v >= 9:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if col != TIE else INK)
            left += v
        lo, hi = c["win_rate_ci95"]
        mark, col = ("▲ push", WIN) if lo > 0.5 else (("▼ SemStamp", LOSS) if hi < 0.5 else ("", INK2))
        ax.text(102, y, f"{c['win_rate']:.3f} [{lo:.3f}, {hi:.3f}] {mark}", va="center", fontsize=7.5, color=col, clip_on=False)
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 5.6)
    ax.set_yticks([5 - k for k in range(len(models))]); ax.set_yticklabels(models, color=INK2, fontsize=8)
    ax.axvline(50, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.set_title(tname, fontsize=9.5, fontweight="bold", color=INK, loc="left")


def main():
    src, out = sys.argv[1], sys.argv[2]
    data = json.load(open("runs/h2h_semstamp.json"))
    by = {(r["domain"], r["model"]): r[src] for r in data}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "text.color": INK})
    fig, axes = plt.subplots(6, 2, figsize=(16, 15), facecolor=SURF)
    for i, (d, name) in enumerate(LEFT):
        panel(axes[i, 0], {m: by.get((d, m)) for m in BASE_CHAT}, BASE_CHAT, name, src)
    for i, (d, name) in enumerate(RIGHT):
        panel(axes[i, 1], {m: by.get((d, m)) for m in CHAT}, CHAT if d != "sp_mbpp" else CHAT, name, src)
    for ax in axes[-1]:
        ax.set_xlabel("% of prompts", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WIN, TIE, LOSS)]
    fig.legend(handles, ["Push + Margin better", "tie", "SemStamp better"], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10)
    fig.suptitle(TITLE[src], y=1.0, fontsize=11.5)
    fig.text(0.5, 0.0, "Right of each bar: push win rate = push better + ½ tie, bootstrap 95% CI over prompts; ▲/▼ = CI entirely above/below 0.5. "
             "n = 1,000 prompts per bar (ours 500; GSM8K 1,319; IFEval 541; MBPP 974).\n" + NOTE[src], ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.075, right=0.86, top=0.94, bottom=0.05, hspace=0.55, wspace=0.62)
    fig.savefig(out, dpi=160, facecolor=SURF, bbox_inches="tight", pad_inches=0.2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
