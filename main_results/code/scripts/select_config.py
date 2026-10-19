"""Apply the pre-registered selection rules (docs/preregistration.md) mechanically; writes runs/selection.json.

Usage: python scripts/select_config.py --dev runs/detector_variants_dev.json --tradeoff runs/detector_variants_tradeoff.json \
           --judge runs/tradeoff/eval/api_judge_google_gemma-4-31b-it.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_GAIN_OVER_T = 0.05
MAX_DEV_FPR = 0.02
MIN_WIN_RATE = 0.40
MAX_MAJOR_FLAW_INCREASE = 0.10
FALLBACK_MIN_TPR = 0.50


def choose_detector(dev: dict) -> dict:
    """Amended rule (docs/preregistration.md, amendment 2): false positives are judged on the POOLED clean arms."""
    table = {}
    for v in next(iter(dev["runs"].values()))["variants"]:
        tprs, fp, pairs, per_run = [], 0, 0, []
        for run in dev["runs"].values():
            for arm, r in run["variants"][v].items():
                if "tpr" in r:
                    tprs.append(r["tpr"])
                if arm == "base" and "fpr" in r:
                    n = int(r.get("n_pairs") or 0)
                    fp += int(r["fp"]) if "fp" in r else int(round(r["fpr"] * n))
                    pairs += n
                    per_run.append(r["fpr"])
        table[v] = {"mean_tpr": float(np.mean(tprs)), "pooled_base_fpr": (fp / pairs) if pairs else None,
                    "max_base_fpr": float(max(per_run)) if per_run else None, "n_arms": len(tprs)}
    ok = {v: t for v, t in table.items() if t["pooled_base_fpr"] is not None and t["pooled_base_fpr"] <= MAX_DEV_FPR}
    best = max(ok, key=lambda v: ok[v]["mean_tpr"]) if ok else "T"
    chosen = best if ("T" not in ok or ok[best]["mean_tpr"] >= ok["T"]["mean_tpr"] + MIN_GAIN_OVER_T) else "T"
    ok_orig = {v: t for v, t in table.items() if t["max_base_fpr"] is not None and t["max_base_fpr"] <= MAX_DEV_FPR}
    best_orig = max(ok_orig, key=lambda v: ok_orig[v]["mean_tpr"]) if ok_orig else "T"
    orig = best_orig if best_orig != "T" and table[best_orig]["mean_tpr"] >= table["T"]["mean_tpr"] + MIN_GAIN_OVER_T else "T"
    return {"chosen": chosen, "best_by_tpr": best, "original_rule_outcome": orig, "table": table}


def choose_arm(tradeoff: dict, judge: dict, detector: str) -> dict:
    run = next(iter(tradeoff["runs"].values()))
    rows = {}
    for arm, r in run["variants"][detector].items():
        if "tpr" not in r:
            continue
        j = judge["per_arm"].get(arm, {})
        rows[arm] = {"tpr": r["tpr"], "tpr_ci": r["tpr_ci"], "auroc": r["auroc"], "win_rate": j.get("win_rate"),
                     "major_flaw_delta": j.get("major_flaw_delta"), "on_task_delta": j.get("on_task_delta")}
        rows[arm]["passes_quality_bar"] = bool(j and j["win_rate"] >= MIN_WIN_RATE and j["major_flaw_delta"] <= MAX_MAJOR_FLAW_INCREASE)
    good = [a for a, r in rows.items() if r["passes_quality_bar"]]
    if good:
        chosen, how = max(good, key=lambda a: rows[a]["tpr"]), "highest TPR among arms that pass the quality bar"
    else:
        cand = [a for a, r in rows.items() if r["tpr"] >= FALLBACK_MIN_TPR and r["win_rate"] is not None]
        chosen = max(cand, key=lambda a: rows[a]["win_rate"]) if cand else max(rows, key=lambda a: rows[a]["tpr"])
        how = "NO arm passed the quality bar: fallback rule (report this as the headline)"
    return {"chosen": chosen, "how": how, "table": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--tradeoff", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", default="runs/selection.json")
    a = ap.parse_args()
    det = choose_detector(json.load(open(a.dev)))
    arm = choose_arm(json.load(open(a.tradeoff)), json.load(open(a.judge)), det["chosen"])
    out = {"detector": det, "configuration": arm, "rules": "docs/preregistration.md"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"detector: {det['chosen']} (amended rule) | original rule: {det['original_rule_outcome']}")
    for v, t in sorted(det["table"].items(), key=lambda kv: -kv[1]["mean_tpr"]):
        print(f"   {v:18s} mean TPR={t['mean_tpr']:.3f} pooled clean FPR={t['pooled_base_fpr']:.4f} (max per run {t['max_base_fpr']:.4f})")
    print(f"configuration: {arm['chosen']} -- {arm['how']}")
    for k, r in sorted(arm["table"].items(), key=lambda kv: -kv[1]["tpr"]):
        print(f"   {k:12s} TPR={r['tpr']:.2f} AUROC={r['auroc']:.3f} win={r['win_rate']} major-flaw delta={r['major_flaw_delta']} pass={r['passes_quality_bar']}")


if __name__ == "__main__":
    main()
