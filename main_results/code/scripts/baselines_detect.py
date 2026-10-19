"""Detection for the reference watermarks, calibrated the way ours is: 1% FPR on an independent clean null.

Each paper ships its own fixed threshold (KGW z=4, SemStamp z=...).  Those are not comparable to our numbers,
which come from an empirical upper-tail threshold on clean text.  Here every method is given the same treatment:
score a large clean cohort, take the conservative 1% rank threshold, then report TPR on the marked texts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld.config import load_config  # noqa: E402
from wmworld.posstats import auroc, upper_tail_threshold, wilson  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

ALPHA = 0.01
EMBEDDER = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"


def texts_of(run: str, arms: set[str] | None = None, min_tokens: int = 32) -> list[dict]:
    gens, _ = collect_run_texts(Path(run))
    return [g for g in gens if g.get("status", "ok") == "ok" and g.get("text")
            and int(g.get("visible_tokens") or 0) >= min_tokens and (arms is None or g["arm"] in arms)]


def kgw_scores(texts: list[str], scheme: str, device: str) -> list[float | None]:
    sys.path.insert(0, str(Path("third_party/lm-watermarking").resolve()))
    from extended_watermark_processor import WatermarkDetector
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")
    det = WatermarkDetector(vocab=list(tok.get_vocab().values()), gamma=0.25, seeding_scheme=scheme,
                            device=torch.device(device), tokenizer=tok, z_threshold=4.0,
                            normalizers=[], ignore_repeated_ngrams=True)
    out = []
    for i, t in enumerate(texts):
        try:
            out.append(float(det.detect(t)["z_score"]))
        except Exception:
            out.append(None)
        if (i + 1) % 200 == 0:
            print(f"  kgw scored {i + 1}/{len(texts)}", flush=True)
    return out


def semstamp_scores(texts: list[str], lsh_dim: int, lmbd: float, device: str) -> list[float | None]:
    sys.path.insert(0, str(Path("third_party/SemStamp").resolve()))
    from detection_utils import detect_lsh
    from sbert_lsh_model import SBERTLSHModel
    lsh = SBERTLSHModel(device=device, batch_size=32, lsh_dim=lsh_dim, sbert_type="base", lsh_model_path=EMBEDDER)
    out = []
    for i, t in enumerate(texts):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
        try:
            out.append(float(detect_lsh(sents, lsh, lmbd, lsh_dim)) if len(sents) >= 3 else None)
        except Exception:
            out.append(None)
        if (i + 1) % 100 == 0:
            print(f"  semstamp scored {i + 1}/{len(texts)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["kgw", "semstamp"], required=True)
    ap.add_argument("--marked", action="append", required=True, help="run dir(s) of watermarked text")
    ap.add_argument("--null", action="append", default=["runs/null_calibration", "runs/conf_cross"],
                    help="clean run dirs (base arm only) for the threshold")
    ap.add_argument("--seeding-scheme", default="simple_1")
    ap.add_argument("--lsh-dim", type=int, default=3)
    ap.add_argument("--lmbd", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    null_texts = [g for r in a.null for g in texts_of(r, arms={"base"})]
    print(f"clean null: {len(null_texts)} texts from {a.null}", flush=True)
    score = (lambda ts: kgw_scores(ts, a.seeding_scheme, a.device)) if a.method == "kgw" else \
            (lambda ts: semstamp_scores(ts, a.lsh_dim, a.lmbd, a.device))
    null = [z for z in score([g["text"] for g in null_texts]) if z is not None and np.isfinite(z)]
    thr = upper_tail_threshold(null, ALPHA)
    res = {"method": a.method, "alpha": ALPHA, "threshold": thr, "null": {"n": len(null), "mean": float(np.mean(null)),
           "sd": float(np.std(null, ddof=1)), "runs": a.null}, "arms": {}}
    print(f"null: n={len(null)} mean={np.mean(null):+.3f} sd={np.std(null, ddof=1):.3f} -> 1% threshold z>{thr:.3f}", flush=True)
    for run in a.marked:
        gens = texts_of(run)
        for arm in sorted({g["arm"] for g in gens}):
            sub = [g for g in gens if g["arm"] == arm]
            z = score([g["text"] for g in sub])
            ok = [x for x in z if x is not None and np.isfinite(x)]
            hits = sum(x > thr for x in ok)
            lo, hi = wilson(hits, len(ok))
            res["arms"][f"{Path(run).name}/{arm}"] = {
                "n": len(ok), "n_texts": len(sub), "tpr": hits / max(1, len(ok)), "tpr_ci95": [lo, hi],
                "auroc": auroc(ok, null), "z_mean": float(np.mean(ok)),
                "median_tokens": float(np.median([g["visible_tokens"] for g in sub]))}
            r = res["arms"][f"{Path(run).name}/{arm}"]
            print(f"{Path(run).name}/{arm}: n={r['n']:4d} z={r['z_mean']:+.2f} TPR@1%={r['tpr']:.3f} "
                  f"[{lo:.3f},{hi:.3f}] AUROC={r['auroc']:.3f} median tokens={r['median_tokens']:.0f}", flush=True)
    write_json(Path(a.out), res)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
