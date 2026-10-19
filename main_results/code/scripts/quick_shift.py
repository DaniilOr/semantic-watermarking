"""Preliminary per-arm numbers from a run whose scoring is still in progress (assigned-key scores only).

Usage: python scripts/quick_shift.py runs/<name> [runs/<name> ...]
Prints, per arm: n, mean score under the assigned key, paired shift vs base (same prompt), mean t-statistic,
mean visible tokens, EOS rate, mean beta_t and steered fraction from the private trajectories (PoE arms).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main(runs: list[str]) -> None:
    for run in runs:
        rd = Path(run)
        gens = [g for f in sorted(rd.glob("generations*.jsonl")) for g in load(f)]
        dets = {(r["text_hash"], r["key_id"]): r for r in load(rd / "detections_raw.jsonl")}
        traj = {}
        for t in [t for f in sorted(rd.glob("trajectories*.jsonl")) for t in load(f)]:
            c = t.get("cost") or {}
            if "mean_beta_t" in c:
                traj[t["output_id"]] = c
        by_arm: dict[str, list[dict]] = {}
        for g in gens:
            d = dets.get((g["text_hash"], g.get("assigned_key_id")))
            by_arm.setdefault(g["arm"], []).append({"g": g, "d": d, "c": traj.get(g["id"])})
        base = {x["g"]["prompt_id"]: x for x in by_arm.get("base", [])}
        print(f"== {run}: {len(gens)} generations, {len(dets)} detection records")
        print(f"{'arm':14s} {'n':>3s} {'scored':>6s} {'score':>7s} {'shift':>7s} {'pos%':>5s} {'t_mean':>7s} {'tokens':>6s} {'eos%':>5s} {'beta_t':>7s} {'steered%':>8s} {'final_t':>7s}")
        for arm, xs in by_arm.items():
            sc = [x["d"]["score"] for x in xs if x["d"] and x["d"].get("score") is not None]
            shifts = []
            for x in xs:
                b = base.get(x["g"]["prompt_id"])
                if x["d"] and b and b["d"] and x["d"].get("score") is not None and b["d"].get("score") is not None:
                    shifts.append(x["d"]["score"] - b["d"]["score"])
            ts = [x["d"]["t_stat"] for x in xs if x["d"] and x["d"].get("t_stat") is not None]
            toks = [x["g"]["visible_tokens"] for x in xs]
            eos = sum(1 for x in xs if x["g"].get("stop_reason") == "eos")
            bt = [x["c"]["mean_beta_t"] for x in xs if x["c"]]
            st = [x["c"]["n_steered_tokens"] / max(1, x["c"]["n_tokens"]) for x in xs if x["c"]]
            ft = [x["c"]["final_t_run"] for x in xs if x["c"] and x["c"].get("final_t_run") is not None]
            m = lambda v: (sum(v) / len(v)) if v else float("nan")
            print(f"{arm:14s} {len(xs):3d} {len(sc):6d} {m(sc):7.3f} {m(shifts):7.3f} {100 * m([s > 0 for s in shifts]):5.0f} {m(ts):7.2f} "
                  f"{m(toks):6.0f} {100 * eos / max(1, len(xs)):5.0f} {m(bt):7.2f} {100 * m(st):8.0f} {m(ft):7.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
