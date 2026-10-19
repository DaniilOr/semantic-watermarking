"""Segment-level key contrast vs the token-level detector, on clean texts and their paraphrases.

A paraphrase keeps WHAT a passage says and changes HOW it says it.  The current detector reads the contrast of
each realised token in the context of everything before it, so a rewrite resamples every term of the sum.  Two
changes are separated here, over a sweep of segment lengths:

  in-context (ic)   the per-token contrast averaged inside each segment, then tested across segments (free:
                    it is a re-aggregation of the existing dumps).  Still conditioned on the whole preceding
                    text, so a rewrite anywhere upstream moves it.
  independent (ind) each segment scored on its own in the keyed / neutral context with NO preceding text, so a
                    segment whose content survives the rewrite keeps its score whatever happened around it.

Segment length is swept because it trades two measured effects against each other: the key's influence on the
reference model is largest in the first tokens it sees and decays with visible text (0.10 at 0-32 tokens, 0.05 at
64-128, 0.03 at 128-256), which favours SHORT segments; but a short segment carries little content, so whatever
key preference it shows is mostly about wording, which is exactly what a paraphrase destroys.  'whole' is the
original specification's F_k, 'sent' is sentence boundaries.

Everything is recalibrated on the clean texts OF THE SAME CONDITION, so the comparison is about effect size and
not about thresholds frozen on unattacked text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.posstats import Dump, auroc, upper_tail_threshold  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import collect_run_texts, load_key_records  # noqa: E402
from wmworld.textnorm import normalize_text  # noqa: E402

MIN_SEG, ALPHA = 8, 0.01
ARMS = ("base", "wm_c4")
SEGMENTATIONS = ["whole", "sent", "L32", "L64", "L128"]
CONDITIONS = [("clean", "runs/conf_cross"), ("para_light", "runs/conf_cross_para_light"),
              ("para_strong", "runs/conf_cross_para_strong")]


def spans_for(name: str, text: str, toks: list[int], tokenize) -> list[tuple[int, int]]:
    n = len(toks)
    if name == "whole":
        return [(0, n)] if n >= MIN_SEG else []
    if name.startswith("L"):
        L = int(name[1:])
        out = [(lo, min(lo + L, n)) for lo in range(0, n, L)]
        return [(a, b) for a, b in out if b - a >= MIN_SEG]
    t = normalize_text(text)
    ends, pos = [], 0
    for part in re.split(r"(?<=[.!?])\s+", t.strip()):
        j = t.find(part, pos)
        if j < 0:
            continue
        pos = j + len(part)
        ends.append(min(n, len(tokenize(t[:pos]))))
    out, lo = [], 0
    for e in sorted({x for x in ends if x > 0} | {n}):
        if e - lo >= MIN_SEG:
            out.append((lo, e)); lo = e
    return out


def aggregate(v: np.ndarray, w: np.ndarray) -> dict:
    """Per-segment, per-token contrasts v with segment lengths w -> the aggregations a detector could use."""
    m = v.size
    if m == 0:
        return {}
    out = {"mean": float((v * w).sum() / w.sum())}
    if m >= 3:
        sd = float(v.std(ddof=1))
        if sd > 0:
            out["t"] = float(v.mean() / (sd / np.sqrt(m)))
        out["sign"] = float((v > 0).mean())
    return out


def main() -> None:
    global SEGMENTATIONS, CONDITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="", help="name=run_dir,... (default: clean + the two paraphrase runs)")
    ap.add_argument("--segmentations", default="", help="comma list from whole,sent,L<n>")
    ap.add_argument("--out", default="runs/segment_contrast.json")
    a = ap.parse_args()
    if a.conditions:
        CONDITIONS = [tuple(x.split("=", 1)) for x in a.conditions.split(",")]
    if a.segmentations:
        SEGMENTATIONS = a.segmentations.split(",")
    cfg = load_config("configs/pilot.yaml")
    cfg.validate(require_revision=True)
    ref = build_reference(cfg)
    scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
    keys = load_key_records(cfg, "confirmation")
    key_ids = [k.key_id for k in keys]
    specs = {k.key_id: k.to_keyspec() for k in keys}
    ctx_key = {kid: scorer.context_ids(ks, neutral=False) for kid, ks in specs.items()}
    ctx_neutral = scorer.context_ids(specs[key_ids[0]], neutral=True)
    ctx_max = max(len(c) for c in ctx_key.values())
    print(f"reference {ref.info['model_id']} | {len(key_ids)} keys | segmentations {SEGMENTATIONS}", flush=True)

    report, vectors, parents = {}, defaultdict(dict), {}
    for cond, run in CONDITIONS:
        t0 = time.time()
        gens, _ = collect_run_texts(Path(run))
        gens = [g for g in gens if g.get("status", "ok") == "ok" and g["arm"] in ARMS and g.get("text")]
        dump = Dump(Path(run) / "eval" / "posstats_confirmation.npz")
        d_idx = {g: i for i, g in enumerate(dump.gen_id)}
        for g in gens:
            if g.get("parent_id"):
                parents[(cond, g["id"])] = g["parent_id"]

        texts, spans_of = [], {}
        flat: list[list[int]] = []
        where: dict[tuple[str, str], list[int]] = defaultdict(list)
        for g in gens:
            toks = scorer.tokenize(g["text"])
            if len(toks) < 32 or ctx_max + len(toks) > ref.max_input_tokens:
                continue
            per_seg = {nm: spans_for(nm, g["text"], toks, scorer.tokenize) for nm in SEGMENTATIONS}
            if not per_seg["whole"]:
                continue
            texts.append(g); spans_of[g["id"]] = per_seg
            for nm, sp in per_seg.items():
                for lo, hi in sp:
                    where[(nm, g["id"])].append(len(flat)); flat.append(toks[lo:hi])
        n_tok = np.array([len(t) for t in flat], dtype=np.float64)
        print(f"[{cond}] {len(texts)}/{len(gens)} texts, {len(flat)} segments, {int(n_tok.sum())} tokens per context", flush=True)

        ell_neu = np.array(ref.logprob_sums(ctx_neutral, flat))
        per_key = {}
        for n_done, kid in enumerate(key_ids, 1):
            per_key[kid] = (np.array(ref.logprob_sums(ctx_key[kid], flat)) - ell_neu) / n_tok
            print(f"  [{cond}] key {n_done}/{len(key_ids)} ({time.time() - t0:.0f}s)", flush=True)

        S: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for g in texts:
            gid = g["id"]
            di = d_idx.get(gid)
            off, n_ref = (int(dump.offsets[di]), dump.n_tokens(di)) if di is not None else (None, 0)
            for kid in key_ids:
                if di is not None and kid in dump.kidx:
                    S["tok_T_clip05_gate"][kid][gid] = dump.stat("T_clip05_gate", di, kid)
                    S["tok_T"][kid][gid] = dump.stat("T", di, kid)
                for nm in SEGMENTATIONS:
                    idx = where[(nm, gid)]
                    if idx:
                        for agg, val in aggregate(per_key[kid][idx], n_tok[idx]).items():
                            S[f"ind_{nm}_{agg}"][kid][gid] = val
                    if off is None or kid not in dump.kidx:
                        continue
                    k = dump.kidx[kid]
                    sp = [(lo, min(hi, n_ref)) for lo, hi in spans_of[gid][nm] if min(hi, n_ref) - lo >= MIN_SEG]
                    if sp:
                        v = np.array([dump.d[k, off + lo:off + hi].mean() for lo, hi in sp])
                        w = np.array([hi - lo for lo, hi in sp], dtype=np.float64)
                        for agg, val in aggregate(v, w).items():
                            S[f"ic_{nm}_{agg}"][kid][gid] = val

        arm_of = {g["id"]: g["arm"] for g in texts}
        assigned = {g["id"]: g.get("assigned_key_id") for g in texts}
        report[cond] = {}
        for name, by_key in S.items():
            null = {k: [v for gid, v in by_key[k].items() if arm_of[gid] == "base" and v is not None] for k in key_ids}
            if any(len(null[k]) < 30 for k in key_ids):
                continue
            mu = {k: float(np.mean(null[k])) for k in key_ids}
            sd = {k: float(np.std(null[k], ddof=1)) or 1.0 for k in key_ids}
            z_null = [(x - mu[k]) / sd[k] for k in key_ids for x in null[k] if np.isfinite(x)]
            z_pos = [(by_key[assigned[gid]][gid] - mu[assigned[gid]]) / sd[assigned[gid]]
                     for gid in arm_of if arm_of[gid] == "wm_c4" and assigned[gid] in by_key
                     and by_key[assigned[gid]].get(gid) is not None and np.isfinite(by_key[assigned[gid]][gid])]
            if len(z_pos) < 30:
                continue
            thr = upper_tail_threshold(z_null, ALPHA)
            report[cond][name] = {"n": len(z_pos), "n_null": len(z_null), "tpr": float(np.mean([z > thr for z in z_pos])),
                                  "auroc": auroc(z_pos, z_null), "z_mean": float(np.mean(z_pos))}
            vectors[name][cond] = {gid: [by_key[k].get(gid) for k in key_ids] for gid in arm_of}
        json.dump({"detection": report, "partial": True}, open(a.out, "w"), indent=1)
        print(f"[{cond}] done, {len(report[cond])} statistics written ({time.time() - t0:.0f}s)", flush=True)

    stab = {}
    for name, by_cond in vectors.items():
        if "clean" not in by_cond:
            continue
        stab[name] = {}
        for cond in ("para_light", "para_strong"):
            cors = []
            for gid, vec in by_cond.get(cond, {}).items():
                base = by_cond["clean"].get(parents.get((cond, gid)))
                if not base:
                    continue
                a, b = np.array(vec, dtype=np.float64), np.array(base, dtype=np.float64)
                if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and a.std() > 0 and b.std() > 0:
                    cors.append(float(np.corrcoef(a, b)[0, 1]))
            if cors:
                stab[name][cond] = {"n": len(cors), "mean_corr": float(np.mean(cors))}

    print("\n" + "=" * 112)
    print("Detection, recalibrated on the clean texts of each condition (per-key standardisation, pooled 1% threshold)")
    print(f"{'statistic':26s} " + "".join(f"{c:>24s}" for c, _ in CONDITIONS) + f"{'key-vector corr':>22s}")
    print(f"{'':26s} " + "".join(f"{'TPR    AUROC':>24s}" for _ in CONDITIONS) + f"{'light  strong':>22s}")
    for pre in ("tok", "ic", "ind"):
        names = [n for n in sorted(report[next(iter(report))]) if n.startswith(pre + "_")]
        names.sort(key=lambda n: (SEGMENTATIONS.index(n.split("_")[1]) if n.split("_")[1] in SEGMENTATIONS else -1, n))
        if names:
            print("-" * 112)
        for name in names:
            row = f"{name:26s} "
            for cond, _ in CONDITIONS:
                r = report[cond].get(name)
                row += f"{r['tpr']:15.3f} {r['auroc']:8.3f}" if r else f"{'-':>24s}"
            s = stab.get(name, {})
            row += f"{s.get('para_light', {}).get('mean_corr', float('nan')):14.2f}{s.get('para_strong', {}).get('mean_corr', float('nan')):8.2f}"
            print(row)
    json.dump({"detection": report, "stability": stab}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
