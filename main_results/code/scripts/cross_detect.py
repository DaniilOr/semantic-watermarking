"""Cross-model detection: score a run's texts with the REFERENCE model under every target key.

Detection with a second model needs no vocabulary bridge: the text is simply teacher-forced under the reference
model's keyed and neutral contexts, in its own tokenization.  Reports, per arm, the within-text statistic
T = mean(d)/(sd/sqrt(n)), the pooled clean null (base texts x all keys), the true-positive rate at the 1% threshold
and the false-attribution rate under wrong keys.

Usage: python scripts/cross_detect.py --config configs/pilot.yaml --run runs/<name> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.records import read_json, write_json  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import collect_run_texts, load_key_records  # noqa: E402


def tstat(d):
    if not d or len(d) < 2:
        return None
    a = np.asarray(d, dtype=np.float64)
    sd = float(a.std(ddof=1))
    return float(a.mean() / (sd / math.sqrt(a.size))) if sd > 0 else None


def auroc(pos, neg):
    if not pos or not neg:
        return None
    p, n = np.asarray(pos), np.asarray(neg)
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--decoys", type=int, default=0, help="also compute the decoy-standardized statistic with N decoy phrases")
    ap.add_argument("--bank", default="decoy_rw")
    ap.add_argument("--prefixes", default="", help="comma list of token prefix lengths, e.g. 128,256,512,1024")
    ap.add_argument("--null", default="", help="frozen null artifact from scripts/freeze_null.py (per-key thresholds)")
    ap.add_argument("--span", type=int, default=0, help="span-aware scan: per-span statistic, report the largest spans")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.validate(require_revision=True)
    ref = build_reference(cfg)
    if args.device:
        ref.cfg.device = args.device
    scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
    print(f"reference world: {json.dumps(ref.info)}", flush=True)
    for run in args.run:
        t0 = time.time()
        rd = Path(run)
        manifest = read_json(rd / "manifest.json")
        keys = load_key_records(cfg, manifest.get("key_set") or "randomwords")
        specs = {k.key_id: k.to_keyspec() for k in keys}
        gens, _ = collect_run_texts(rd)
        gens = [g for g in gens if g.get("status", "ok") == "ok" and g.get("visible_tokens", 0) >= 32]
        texts = {g["text_hash"]: g["text"] for g in gens}
        hashes = sorted(texts)
        t_all = {}
        for kid, ks in specs.items():
            for s in range(0, len(hashes), 32):
                chunk = hashes[s:s + 32]
                for h, d in zip(chunk, scorer.per_token_contrasts([texts[h] for h in chunk], ks)):
                    t_all[(h, kid)] = tstat(d)
            print(f"  [{rd.name}] key {kid} scored ({time.time() - t0:.0f}s)", flush=True)
        # decoy-standardized statistic: z = (T_key - mean_d T_d) / sd_d T_d, computed per text with the SAME model.
        # It removes each text's own sensitivity to "any background phrase", which is what makes a base model's
        # clean contrast distribution heavy-tailed.
        z_all = {}
        if args.decoys:
            from wmworld.keys import load_keys
            from wmworld.stages import keys_path
            bank = load_keys(read_json(keys_path(cfg, args.bank)))[: args.decoys]
            t_dec = defaultdict(dict)
            for dk in bank:
                ks = dk.to_keyspec()
                for s0 in range(0, len(hashes), 32):
                    chunk = hashes[s0:s0 + 32]
                    for h, d in zip(chunk, scorer.per_token_contrasts([texts[h] for h in chunk], ks)):
                        t_dec[h][dk.key_id] = tstat(d)
            print(f"  [{rd.name}] {len(bank)} decoy phrases scored ({time.time() - t0:.0f}s)", flush=True)
            for h in hashes:
                vals = [v for v in t_dec[h].values() if v is not None]
                if len(vals) < 3:
                    continue
                m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
                for kid in specs:
                    tv = t_all.get((h, kid))
                    if tv is not None and sd > 0:
                        z_all[(h, kid)] = (tv - m) / sd
        by_arm = defaultdict(list)
        for g in gens:
            by_arm[g["arm"]].append(g)
        # The null: by default this run's own base arm, which is only ~60 INDEPENDENT texts -- its 1% tail rests on
        # about six observations and swings badly between runs (2.60 vs 3.85 on two runs of one configuration, which
        # moved a fixed T=3.2 from 87% detected to 5%).  --null supplies a frozen null over the dedicated partition.
        frozen = read_json(Path(args.null)) if args.null else None
        null = [t_all[(g["text_hash"], k)] for g in by_arm.get("base", []) for k in specs if t_all.get((g["text_hash"], k)) is not None]
        arr = np.sort(np.asarray(null))
        kq = int(math.floor(0.01 * (len(arr) + 1))) - 1
        thr_run = float(arr[len(arr) - 1 - kq]) if kq >= 0 else float("inf")
        if frozen:
            thr = float(frozen["pooled"]["threshold"])
            thr_key = {k: float(v["threshold"]) for k, v in frozen["per_key"].items()}
            print(f"frozen null: {frozen['n_texts']} texts x {len(frozen['per_key'])} keys from {frozen['null_run']} "
                  f"-> pooled T>{thr:.2f}; this run's own base arm would have given T>{thr_run:.2f}")
        else:
            thr, thr_key = thr_run, {}
            print(f"clean null n={len(null)} mean={np.mean(null):.2f} sd={np.std(null, ddof=1):.2f} -> 1% threshold T>{thr:.2f}")
        tk = lambda kid: thr_key.get(kid, thr)          # per-key threshold when frozen, else the pooled one
        out = {"run": str(rd), "reference": ref.info, "threshold_alpha0.01": thr, "null_n": len(null),
               "threshold_source": ("frozen:" + args.null) if frozen else "run_base_arm",
               "threshold_run_base_arm": thr_run, "per_arm": {}}
        for arm, gs in by_arm.items():
            pos = [t_all[(g["text_hash"], g["assigned_key_id"])] for g in gs if t_all.get((g["text_hash"], g.get("assigned_key_id"))) is not None]
            wrong = [t_all[(g["text_hash"], k)] for g in gs for k in specs
                     if k != g.get("assigned_key_id") and t_all.get((g["text_hash"], k)) is not None]
            # threshold applied PER KEY when a frozen null is available (keys do not share a null distribution)
            pos_f = [(t_all[(g["text_hash"], g["assigned_key_id"])], g["assigned_key_id"]) for g in gs
                     if t_all.get((g["text_hash"], g.get("assigned_key_id"))) is not None]
            wrong_f = [(t_all[(g["text_hash"], k)], k) for g in gs for k in specs
                       if k != g.get("assigned_key_id") and t_all.get((g["text_hash"], k)) is not None]
            rec = {"n": len(pos), "t_mean": float(np.mean(pos)) if pos else None,
                   "tpr_alpha0.01": float(np.mean([v > tk(k) for v, k in pos_f])) if pos_f else None,
                   "false_attribution_alpha0.01": float(np.mean([v > tk(k) for v, k in wrong_f])) if wrong_f else None,
                   "auroc_vs_clean": auroc(pos, null)}
            if z_all:
                zn = [z_all[(g["text_hash"], k)] for g in by_arm.get("base", []) for k in specs if (g["text_hash"], k) in z_all]
                za = np.sort(np.asarray(zn))
                kq2 = int(math.floor(0.01 * (len(za) + 1))) - 1
                zthr = float(za[len(za) - 1 - kq2]) if kq2 >= 0 else float("inf")
                zpos = [z_all[(g["text_hash"], g["assigned_key_id"])] for g in gs if (g["text_hash"], g.get("assigned_key_id")) in z_all]
                zwrong = [z_all[(g["text_hash"], k)] for g in gs for k in specs
                          if k != g.get("assigned_key_id") and (g["text_hash"], k) in z_all]
                rec["decoy_z"] = {"threshold": zthr, "null_n": len(zn), "z_mean": float(np.mean(zpos)) if zpos else None,
                                  "tpr_alpha0.01": float(np.mean([p > zthr for p in zpos])) if zpos else None,
                                  "false_attribution_alpha0.01": float(np.mean([w > zthr for w in zwrong])) if zwrong else None,
                                  "auroc_vs_clean": auroc(zpos, zn)}
                dz = rec["decoy_z"]
                print(f"{rd.name:22s} {arm:12s} DECOY-Z  z={dz['z_mean'] if zpos else float('nan'):6.2f} thr={zthr:5.2f} "
                      f"TPR@1%={dz['tpr_alpha0.01'] if zpos else float('nan'):.2f} "
                      f"FA@1%={dz['false_attribution_alpha0.01'] if zwrong else float('nan'):.2f} "
                      f"AUROC={dz['auroc_vs_clean'] or float('nan'):.3f}", flush=True)
            out["per_arm"][arm] = rec
            print(f"{rd.name:22s} {arm:12s} n={rec['n']:4d} T={rec['t_mean'] if pos else float('nan'):6.2f} "
                  f"TPR@1%={rec['tpr_alpha0.01'] if pos else float('nan'):.2f} "
                  f"FA@1%={rec['false_attribution_alpha0.01'] if wrong else float('nan'):.2f} "
                  f"AUROC={rec['auroc_vs_clean'] or float('nan'):.3f}", flush=True)
        # ---- span-aware scan: each span is marked on its own, so score spans and keep the largest ----
        if args.span:
            N = args.span
            spans = {}
            for h in hashes:
                toks = scorer.tokenize(texts[h])
                spans[h] = [scorer.backend.decode(toks[i:i + N]) for i in range(0, len(toks) - N + 1, N)]
            flat, index = [], []
            for h in hashes:
                for j, sp in enumerate(spans[h]):
                    flat.append(sp); index.append((h, j))
            sc_span = {}
            for kid, ks in specs.items():
                for s0 in range(0, len(flat), 64):
                    chunk = flat[s0:s0 + 64]
                    for (h, j), d in zip(index[s0:s0 + 64], scorer.per_token_contrasts(chunk, ks)):
                        sc_span[(h, kid, j)] = tstat(d)
            def agg(h, kid, how):
                v = [sc_span[(h, kid, j)] for j in range(len(spans[h])) if sc_span.get((h, kid, j)) is not None]
                if not v:
                    return None
                v.sort(reverse=True)
                return v[0] if how == "max" else float(np.mean(v[:2]))
            out["span_scan"] = {"span_tokens": N, "per_arm": {}}
            for how in ("max", "top2"):
                nl = [agg(g["text_hash"], k, how) for g in by_arm.get("base", []) for k in specs
                      if agg(g["text_hash"], k, how) is not None]
                if len(nl) < 30:
                    continue
                a3 = np.sort(np.asarray(nl)); k3 = int(math.floor(0.01 * (len(a3) + 1))) - 1
                th3 = float(a3[len(a3) - 1 - k3]) if k3 >= 0 else float("inf")
                for arm, gs in by_arm.items():
                    pp = [agg(g["text_hash"], g["assigned_key_id"], how) for g in gs
                          if agg(g["text_hash"], g.get("assigned_key_id"), how) is not None]
                    if not pp:
                        continue
                    r = {"n": len(pp), "stat_mean": float(np.mean(pp)), "threshold": th3,
                         "tpr_alpha0.01": float(np.mean([x > th3 for x in pp])), "auroc_vs_clean": auroc(pp, nl)}
                    out["span_scan"]["per_arm"].setdefault(arm, {})[how] = r
                    print(f"{rd.name:20s} SCAN({N},{how:4s}) {arm:12s} n={r['n']:3d} stat={r['stat_mean']:6.2f} thr={th3:5.2f} "
                          f"TPR@1%={r['tpr_alpha0.01']:.2f} AUROC={r['auroc_vs_clean'] or float('nan'):.3f}", flush=True)
        # ---- detection versus length: T on the first N tokens of the SAME texts, with per-length clean nulls ----
        if args.prefixes:
            lens = [int(x) for x in args.prefixes.split(",") if x.strip()]
            out["by_prefix"] = {}
            for L in lens:
                pre = {}
                for h in hashes:
                    toks = scorer.tokenize(texts[h])
                    pre[h] = scorer.backend.decode(toks[:L]) if len(toks) > L else None   # None: text shorter than L
                use = [h for h in hashes if pre[h] is not None]
                if len(use) < 10:
                    continue
                t_pre = {}
                for kid, ks in specs.items():
                    for s0 in range(0, len(use), 32):
                        chunk = use[s0:s0 + 32]
                        for h, d in zip(chunk, scorer.per_token_contrasts([pre[h] for h in chunk], ks)):
                            t_pre[(h, kid)] = tstat(d)
                nl = [t_pre[(g["text_hash"], k)] for g in by_arm.get("base", []) for k in specs
                      if t_pre.get((g["text_hash"], k)) is not None]
                if len(nl) < 30:
                    continue
                a2 = np.sort(np.asarray(nl)); k2 = int(math.floor(0.01 * (len(a2) + 1))) - 1
                th2 = float(a2[len(a2) - 1 - k2]) if k2 >= 0 else float("inf")
                row = {"n_texts_at_least_this_long": len(use), "null_n": len(nl), "threshold": th2, "per_arm": {}}
                for arm, gs in by_arm.items():
                    pp = [t_pre[(g["text_hash"], g["assigned_key_id"])] for g in gs
                          if t_pre.get((g["text_hash"], g.get("assigned_key_id"))) is not None]
                    if not pp:
                        continue
                    row["per_arm"][arm] = {"n": len(pp), "t_mean": float(np.mean(pp)),
                                           "tpr_alpha0.01": float(np.mean([x > th2 for x in pp])),
                                           "auroc_vs_clean": auroc(pp, nl)}
                    print(f"{rd.name:20s} prefix {L:5d} {arm:12s} n={len(pp):3d} T={row['per_arm'][arm]['t_mean']:6.2f} "
                          f"TPR@1%={row['per_arm'][arm]['tpr_alpha0.01']:.2f} AUROC={row['per_arm'][arm]['auroc_vs_clean'] or float('nan'):.3f}",
                          flush=True)
                out["by_prefix"][str(L)] = row
        (rd / "eval").mkdir(exist_ok=True)
        write_json(rd / "eval" / "cross_detect.json", out)


if __name__ == "__main__":
    main()
