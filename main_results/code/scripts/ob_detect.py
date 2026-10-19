"""Detection on the OpenStamp-style benchmark (amendment 17), per dataset, text + key only.
(Sentence arms are read with z_chain / t_cont, segment arms with z_segchain / t_segcont; all four are reported.)

Statistics (sentence chain, semwm.score_text): z_chain (SemStamp's count) and t_cont (clipped depth).
Per dataset (ob_arxiv, ob_booksum, ob_wikipedia):
  human null  the 2,000 human continuations of that dataset's NULL documents (runs/ob_bench_null) -> thresholds at 1%
              and 0.1% FPR (conservative rank thresholds, wmworld.posstats.upper_tail_threshold)
  model null  the plain 3B's own continuations of the 500 test prompts (arm "base" of runs/ob_bench) -> 1% threshold
Reported per run (clean + attacked copies) x arm x dataset: TPR at each threshold with Wilson intervals, AUROC against
the human null, n.  Clean cohorts (base, clean_seed1, human) report their flag rates, i.e. false-positive rates.
"""

from __future__ import annotations

import argparse
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

DOMAINS = ("ob_arxiv", "ob_booksum", "ob_wikipedia")
STATS = ("z_chain", "t_cont", "z_segchain", "t_segcont")      # sentence chain, then the 16-word segment chain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ob_bench")
    ap.add_argument("--null", default="runs/ob_bench_null")
    ap.add_argument("--attacks", nargs="*", default=["reorder_sent", "resegment", "para_light", "para_strong"])
    ap.add_argument("--embedder", default="/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/ob_detect.json")
    ap.add_argument("--domains", nargs="*", default=list(DOMAINS), help="datasets to score (default: ArXiv, BookSum, Wikipedia)")
    ap.add_argument("--arms", nargs="*", default=None, help="only these arms of --run and its attacked copies (early reads); "
                                                             "keep 'base', it sets the model_1%% threshold")
    a = ap.parse_args()
    t0 = time.time()
    emb = semwm.Embedder(a.embedder, device=a.device, batch_size=256)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
    runs = [a.run] + [f"{a.run}_{x}" for x in a.attacks if Path(f"{a.run}_{x}").exists()]
    texts = {}                                                              # text_hash -> text
    groups = defaultdict(list)                                              # (run, arm, domain) -> [hash]
    for run in runs + [a.null]:
        for g in collect_run_texts(Path(run))[0]:
            if a.arms and run != a.null and g["arm"] not in a.arms:
                continue
            if g.get("status", "ok") == "ok" and g.get("text") and g.get("domain") in a.domains:
                texts.setdefault(g["text_hash"], g["text"]); groups[(Path(run).name, g["arm"], g["domain"])].append(g["text_hash"])
    print(f"{len(texts)} distinct texts in {len(groups)} groups ({time.time() - t0:.0f}s)", flush=True)
    score = {}
    hs = sorted(texts)
    for i in range(0, len(hs), 512):
        chunk = hs[i:i + 512]
        sents = [semwm.split_sentences(texts[h]) for h in chunk]
        segs = [semwm.split_segments(texts[h], 16) for h in chunk]
        flat = emb.encode([s for ss in sents for s in ss] + [s for gg in segs for s in gg])
        k = 0
        es = []
        for ss in sents:
            es.append(flat[k:k + len(ss)]); k += len(ss)
        for h, ss, gg, e in zip(chunk, sents, segs, es):
            eg = flat[k:k + len(gg)]; k += len(gg)
            score[h] = semwm.score_text(texts[h], key, emb, W=16, planes=planes, sent_emb=e, seg_emb=eg)
        if (i // 512) % 10 == 0:
            print(f"  scored {min(i + 512, len(hs))}/{len(hs)} ({time.time() - t0:.0f}s)", flush=True)
    val = lambda h, st: score[h][st]
    report = {"thresholds": {}, "results": {}}
    null_name = Path(a.null).name
    for d in a.domains:
        for st in STATS:
            hv = [x for x in (val(h, st) for h in groups[(null_name, "base", d)]) if x is not None]
            mv = [x for x in (val(h, st) for h in groups[(Path(a.run).name, "base", d)]) if x is not None]
            report["thresholds"][f"{d}:{st}"] = {"human_1%": upper_tail_threshold(hv, 0.01), "human_0.1%": upper_tail_threshold(hv, 0.001),
                                                "model_1%": upper_tail_threshold(mv, 0.01) if mv else None, "n_human": len(hv), "n_model": len(mv),
                                                "_human_values": hv}
    print(f"\n{'run':26s} {'arm':34s} {'dataset':13s} {'n':>4s} " + " ".join(f"{st + ' ' + t:>18s}" for st in STATS for t in ("h1%", "h0.1%", "m1%")) +
          " " + " ".join(f"{'AUROC ' + st:>13s}" for st in STATS))
    for (run, arm, d), hl in sorted(groups.items()):
        if run == null_name:
            continue
        rec = {"n": len(hl)}
        line = f"{run:26s} {arm:34s} {d:13s} {len(hl):4d} "
        for st in STATS:
            v = [x for x in (val(h, st) for h in hl) if x is not None]
            th = report["thresholds"][f"{d}:{st}"]
            for tname in ("human_1%", "human_0.1%", "model_1%"):
                thr = th[tname]
                if thr is None or not np.isfinite(thr) or not v:
                    rec[f"{st}:{tname}"] = None; line += f"{'-':>18s} "; continue
                k = sum(x > thr for x in v); lo, hi = wilson(k, len(v))
                rec[f"{st}:{tname}"] = {"rate": k / len(v), "ci": [lo, hi]}
                line += f"{k / len(v):7.3f} [{lo:.2f},{hi:.2f}] "
            rec[f"{st}:auroc_vs_human"] = auroc(v, th["_human_values"]) if v else None
        line += " ".join(f"{(rec[f'{st}:auroc_vs_human'] or float('nan')):13.3f}" for st in STATS)
        report["results"][f"{run}|{arm}|{d}"] = rec
        print(line, flush=True)
    for k in report["thresholds"]:
        report["thresholds"][k].pop("_human_values")
    write_json(a.out, report)
    print(f"\nwrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
