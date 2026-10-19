"""Text-only detection for the semantic watermark arms: SemStamp chain z, its continuous variant, and the segment
statistic, under the shipped key (and alternative keys as a wrong-key diagnostic).  CPU is enough.

Thresholds: per statistic, the conservative 1% upper tail of the null texts (`--null` runs, base arm).  Reported per
marked run and arm: TPR with a Wilson interval, AUROC against the null, mean statistic, wrong-key flag rate (alt keys
at their own thresholds), and the FPR of every clean cohort (including attacked clean texts) at the same thresholds.
Detections written per run hold text_hash and statistics only -- no prompt, no arm, no label.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.posstats import auroc, upper_tail_threshold, wilson  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

STATS = ("z_chain", "t_cont", "t_seg", "z_segchain", "t_segcont")


def texts_of(run: str, arms: set[str] | None, min_tokens: int) -> list[dict]:
    gens, _ = collect_run_texts(Path(run))
    return [g for g in gens if g.get("status", "ok") == "ok" and g.get("text") and int(g.get("visible_tokens") or 0) >= min_tokens
            and (arms is None or g["arm"] in arms)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marked", action="append", required=True, help="run dir; every non-base arm is evaluated")
    ap.add_argument("--null", action="append", required=True, help="run dir whose base arm is the null")
    ap.add_argument("--embedder", default="/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--embedder-prefix", default="")
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--lmbd", type=float, default=semwm.LMBD)
    ap.add_argument("--margin", type=float, default=semwm.MARGIN)
    ap.add_argument("--clip", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--alt-keys", type=int, default=6)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/detect_semwm.json")
    a = ap.parse_args()

    emb = semwm.Embedder(a.embedder, device=a.device, batch_size=128, prefix=a.embedder_prefix)
    keys = [semwm.Key.shipped()] + [semwm.Key.alt(i) for i in range(1, a.alt_keys + 1)]
    planes = {k.label(): semwm.Hyperplanes.for_key(k, emb.dim) for k in keys}
    t0 = time.time()

    # ---- embed every distinct text once (sentences and segments), then score under every key ---- #
    pool: dict[str, dict] = {}
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)     # run -> (text_hash, arm, kind)
    for kind, runs in (("null", a.null), ("marked", a.marked)):
        for run in runs:
            for g in texts_of(run, {"base"} if kind == "null" else None, a.min_tokens):
                pool.setdefault(g["text_hash"], g["text"])
                groups[run].append((g["text_hash"], g["arm"], kind))
    hashes = sorted(pool)
    print(f"{len(hashes)} distinct texts across {len(groups)} runs; embedding ({time.time() - t0:.0f}s)", flush=True)
    sent_emb, seg_emb = {}, {}
    for s0 in range(0, len(hashes), 64):
        for h in hashes[s0:s0 + 64]:
            sent_emb[h] = emb.encode(semwm.split_sentences(pool[h]))
            seg_emb[h] = emb.encode(semwm.split_segments(pool[h], a.W))
        if (s0 // 64) % 10 == 0:
            print(f"  embedded {min(s0 + 64, len(hashes))}/{len(hashes)} ({time.time() - t0:.0f}s)", flush=True)
    scores: dict[tuple[str, str], dict] = {}
    for h in hashes:
        for k in keys:
            scores[(h, k.label())] = semwm.score_text(pool[h], k, emb, a.lmbd, a.margin, a.clip, a.W, planes[k.label()],
                                                      sent_emb[h], seg_emb[h])
    print(f"scored {len(hashes)} texts x {len(keys)} keys ({time.time() - t0:.0f}s)", flush=True)

    # ---- null thresholds per key and statistic ---- #
    null_hashes = sorted({h for run in a.null for h, arm, kind in groups[run] if arm == "base"})
    null_tab = {}
    for k in keys:
        for st in STATS:
            v = [scores[(h, k.label())][st] for h in null_hashes]
            v = [x for x in v if x is not None and np.isfinite(x)]
            null_tab[(k.label(), st)] = {"thr": upper_tail_threshold(v, a.alpha), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                         "n": len(v), "values": v}
    shipped = keys[0].label()
    print(f"null: {len(null_hashes)} texts; thresholds (shipped key) " + " ".join(f"{st}>{null_tab[(shipped, st)]['thr']:.3f}" for st in STATS), flush=True)

    # ---- per run / arm ---- #
    report = {"null_runs": a.null, "n_null": len(null_hashes), "alpha": a.alpha, "W": a.W, "clip": a.clip, "embedder": a.embedder,
              "thresholds": {st: null_tab[(shipped, st)]["thr"] for st in STATS}, "runs": {}}
    print(f"\n{'run':28s} {'arm':18s} {'n':>4s} " + " ".join(f"{st + ' TPR [CI]':>22s} {'AUROC':>6s} {'wrong':>6s}" for st in STATS))
    for run, items in groups.items():
        by_arm = defaultdict(list)
        for h, arm, kind in items:
            by_arm[arm].append(h)
        rr = {}
        det_path = Path(run) / "eval" / "semwm_detections.jsonl"
        det_path.parent.mkdir(parents=True, exist_ok=True)
        with det_path.open("w") as fh:
            for h in sorted({h for h, _, _ in items}):
                s = scores[(h, shipped)]
                fh.write(json.dumps({"text_hash": h, "key": shipped, "z_chain": s["z_chain"], "t_cont": s["t_cont"], "t_seg": s["t_seg"], "z_segchain": s["z_segchain"], "t_segcont": s["t_segcont"],
                                     "n_sent": s["n_sent"], "n_seg": s["n_seg"]}) + "\n")
        for arm, hs in sorted(by_arm.items()):
            rec = {"n": len(hs)}
            line = f"{Path(run).name:28s} {arm:18s} {len(hs):4d} "
            for st in STATS:
                nt = null_tab[(shipped, st)]
                v = [scores[(h, shipped)][st] for h in hs]
                v = [x for x in v if x is not None and np.isfinite(x)]
                if not v:
                    line += f"{'-':>22s} {'-':>6s} {'-':>6s} "; continue
                hits = sum(x > nt["thr"] for x in v)
                lo, hi = wilson(hits, len(v))
                wrong = [(scores[(h, k.label())][st], null_tab[(k.label(), st)]["thr"]) for h in hs for k in keys[1:]]
                wrong = [(x, thr) for x, thr in wrong if x is not None and np.isfinite(x)]
                wrong_rate = float(np.mean([x > thr for x, thr in wrong])) if wrong else None
                rec[st] = {"rate": hits / len(v), "ci": [lo, hi], "n": len(v), "mean": float(np.mean(v)),
                           "auroc": auroc(v, nt["values"]), "wrong_key_rate": wrong_rate, "threshold": nt["thr"]}
                line += f"{hits / len(v):8.3f} [{lo:.2f},{hi:.2f}] {rec[st]['auroc']:6.3f} {(wrong_rate if wrong_rate is not None else float('nan')):6.3f} "
            rr[arm] = rec
            print(line + ("  (clean cohort: rate = FPR)" if arm in ("base", "clean_seed1") else ""), flush=True)
        report["runs"][Path(run).name] = rr
    write_json(a.out, report)
    print(f"\nwrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
