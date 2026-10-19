"""Cross-run comparison of steering configurations (one row per run/arm) from runs/<name>/eval/results.json.

Usage: python scripts/poe_compare.py runs/abl_poe5c05 runs/abl_poe10c02 ... [--md]

Columns: paired score shift vs base (score units), rank detector (mean fraction of decoys beaten, beats-all-decoys
rate, z-diff flags at toy alpha 0.01, AUROC z-diff), t detector (AUROC, mean t), top-1 key among all target keys,
quality (paired mean log-lik/token delta vs base, degenerate fraction, distinct-2), leakage (key terms, category
vocabulary), length (mean visible tokens, EOS rate).  base rows give the control values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row(run: Path, arm: str, r: dict) -> dict:
    a = r["per_arm"][arm]
    rd = _g(r, "rank_detector", "per_arm", arm, default={}) or {}
    td = _g(r, "t_detector", "per_arm", arm, default={}) or {}
    sp = _g(r, "specificity", arm, default={}) or {}
    ovr = _g(r, "arms", arm, "overrides", default=None)
    fl = rd.get("flag_rate_zdiff_alpha0.01") or {}
    return {
        "run": run.name, "arm": arm, "n": a.get("n"),
        "shift": _g(a, "paired_shift_vs_base", "mean"),
        "shift_pos": a.get("paired_shift_positive_frac"),
        "rank_frac": rd.get("rank_frac_mean"),
        "beats_all": rd.get("top_among_all_keys_frac"),
        "zdiff_flags": f"{fl.get('k', '-')}/{fl.get('n', '-')}",
        "auroc_z": _g(rd, "auroc_zdiff_vs_base", "auroc"),
        "auroc_t": _g(td, "auroc_t_vs_base", "auroc"),
        "t_mean": _g(td, "t_mean", "mean"),
        "top1": sp.get("top1_among_all_keys"),
        "logp": _g(a, "quality_proxy", "mean_logp_task", "mean"),
        "d_logp": _g(a, "quality_proxy", "paired_delta_vs_base", "mean"),
        "degen": _g(a, "quality_proxy", "degenerate_frac"),
        "loop8": _g(a, "quality_proxy", "loop8_frac"),
        "dist2": _g(a, "quality_proxy", "distinct2_mean"),
        "term_leak": _g(a, "leakage", "any_term_frac"),
        "cat_leak": _g(a, "leakage", "category_any_frac"),
        "tokens": a.get("visible_tokens_mean"),
        "eos": (a.get("stop_reasons") or {}).get("eos", 0) / max(1, a.get("n") or 1),
        "overrides": json.dumps(ovr) if ovr else "",
    }


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def main(argv: list[str]) -> None:
    md = "--md" in argv
    runs = [Path(p) for p in argv if not p.startswith("--")]
    rows = []
    for run in runs:
        p = run / "eval" / "results.json"
        if not p.exists():
            print(f"# {run}: no eval/results.json", file=sys.stderr)
            continue
        r = json.loads(p.read_text())
        for arm in r["per_arm"]:
            rows.append(row(run, arm, r))
    cols = ["run", "arm", "n", "shift", "shift_pos", "rank_frac", "beats_all", "zdiff_flags", "auroc_z", "auroc_t", "t_mean",
            "top1", "logp", "d_logp", "degen", "loop8", "dist2", "term_leak", "cat_leak", "tokens", "eos"]
    if md:
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for x in rows:
            print("| " + " | ".join(fmt(x[c]) for c in cols) + " |")
    else:
        w = {c: max(len(c), *(len(fmt(x[c])) for x in rows)) for c in cols} if rows else {c: len(c) for c in cols}
        print("  ".join(c.ljust(w[c]) for c in cols))
        for x in rows:
            print("  ".join(fmt(x[c]).ljust(w[c]) for c in cols))
    for x in rows:
        if x["overrides"]:
            print(f"# {x['run']}/{x['arm']}: {x['overrides']}")


if __name__ == "__main__":
    main(sys.argv[1:])
