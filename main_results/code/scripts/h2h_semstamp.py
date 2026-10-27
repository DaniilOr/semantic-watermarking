"""Push + Margin vs SemStamp head to head, from judgments that already exist (no API calls).

No judge call ever showed the push text and the SemStamp text side by side: the pairwise judge compared each arm
with the plain answer.  Two comparisons can still be built per prompt, on the same prompts:

  pointwise   the 1-5 quality score each text got on its own (api_pointwise).  Per prompt: push wins if its score is
              higher, tie if equal.  Win rate = P(push higher) + 0.5 P(equal).
  via plain   both arms were judged against the SAME plain answer (api_pref).  Per prompt s_arm = mean over the two
              orders of (1 arm better, 0.5 tie, 0 plain better).  Reported: mean(s_push - s_ss) with a paired CI.

95% CIs: bootstrap over prompts (2000 resamples, seed 0).  Output: runs/h2h_semstamp.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SLUG = "google_gemma-4-31b-it"
RUNS = [("Llama-3B chat", "sp_bench_llama3b"), ("Llama-1B chat", "sp_bench_llama1b"), ("Qwen chat", "sp_bench_qwen08b_nl"),
        ("Llama-3B base", "sp_bench_llama3b_base"), ("Llama-1B base", "sp_bench_llama1b_base"), ("Qwen base", "sp_bench_qwen08b_base_nl"),
        ("Llama-3B chat", "ob_bench"), ("Llama-1B chat", "ob_bench_llama1b"), ("Qwen chat", "ob_bench_qwen08b_nl"),
        ("Llama-3B base", "ob_bench_llama3b_base"), ("Llama-1B base", "ob_bench_llama1b_base"), ("Qwen base", "ob_bench_qwen08b_base_nl"),
        ("Llama-3B chat", "it_bench_llama3b"), ("Llama-1B chat", "it_bench_llama1b"), ("Qwen chat", "it_bench_qwen08b")]
PUSH, SS = "chain_reference", "semstamp"


def boot_ci(x: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    b = [float(rng.choice(x, size=x.size, replace=True).mean()) for _ in range(2000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def pointwise(run: str) -> dict[str, dict[str, tuple[str, float]]]:
    """arm -> prompt_id -> (domain, quality); seed 0 only (the arm's own generation)."""
    out: dict[str, dict] = defaultdict(dict)
    p = Path("runs") / run / "eval" / f"api_pointwise_{SLUG}.jsonl"
    for line in open(p):
        r = json.loads(line)
        q = (r.get("judgment") or {}).get("quality")
        if r["arm"] in (PUSH, SS) and r.get("seed", 0) == 0 and isinstance(q, (int, float)):
            out[r["arm"]][r["prompt_id"]] = (r["domain"], float(q))
    return out


def preference(run: str) -> dict[str, dict[str, tuple[str, float]]]:
    """arm -> prompt_id -> (domain, mean over orders of the arm-vs-plain score); prompts with both orders only."""
    by: dict[tuple, list] = defaultdict(list)
    dom = {}
    p = Path("runs") / run / "eval" / f"api_pref_{SLUG}.jsonl"
    for line in open(p):
        r = json.loads(line)
        if r["arm"] not in (PUSH, SS):
            continue
        b = r["judgment"]["better"]
        by[(r["arm"], r["prompt_id"])].append(0.5 if b == "tie" else (1.0 if b == r["wm_is"] else 0.0))
        dom[r["prompt_id"]] = r.get("domain", "")
    out: dict[str, dict] = defaultdict(dict)
    for (arm, pid), v in by.items():
        if len(v) == 2:
            out[arm][pid] = (dom[pid], float(np.mean(v)))
    return out


def compare(a: dict, b: dict, domain: str | None) -> dict | None:
    pids = sorted(p for p in a if p in b and (domain is None or a[p][0] == domain))
    if not pids:
        return None
    x = np.array([a[p][1] for p in pids])
    y = np.array([b[p][1] for p in pids])
    win = np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))
    return {"n": len(pids), "push_mean": float(x.mean()), "ss_mean": float(y.mean()),
            "p_push_higher": float((x > y).mean()), "p_equal": float((x == y).mean()), "p_ss_higher": float((x < y).mean()),
            "win_rate": float(win.mean()), "win_rate_ci95": boot_ci(win),
            "diff": float((x - y).mean()), "diff_ci95": boot_ci(x - y)}


def main() -> None:
    res = []
    for model, run in RUNS:
        pw, pr = pointwise(run), preference(run)
        doms = sorted({d for d, _ in pw[PUSH].values()} | {d for d, _ in pr[PUSH].values()})
        for d in doms + [None]:
            row = {"model": model, "run": run, "domain": d or "all",
                   "pointwise": compare(pw[PUSH], pw[SS], d), "via_plain": compare(pr[PUSH], pr[SS], d)}
            res.append(row)
            a, v = row["pointwise"], row["via_plain"]
            fa = (f"pw win={a['win_rate']:.3f} [{a['win_rate_ci95'][0]:.3f},{a['win_rate_ci95'][1]:.3f}] "
                  f"push>/=/ss> {a['p_push_higher']:.2f}/{a['p_equal']:.2f}/{a['p_ss_higher']:.2f} q {a['push_mean']:.2f}/{a['ss_mean']:.2f} n={a['n']}") if a else "pw -"
            fv = (f"via-plain push {v['push_mean']:.3f} ss {v['ss_mean']:.3f} diff={v['diff']:+.3f} "
                  f"[{v['diff_ci95'][0]:+.3f},{v['diff_ci95'][1]:+.3f}] n={v['n']}") if v else "via-plain -"
            print(f"{run:26s} {model:14s} {row['domain']:18s} | {fa} | {fv}", flush=True)
    Path("runs/h2h_semstamp.json").write_text(json.dumps(res, indent=1))
    print("wrote runs/h2h_semstamp.json")


if __name__ == "__main__":
    main()
