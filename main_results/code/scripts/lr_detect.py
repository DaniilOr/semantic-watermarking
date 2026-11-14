"""Likelihood-ratio (LR) detector on sentence depth, as described in "Push+Margin with a likelihood-ratio detector: results
to date (3 November 2026)", section 1, applied to our large-scale runs (six generators, SemStamp's datasets, our
benchmark, instruction tasks, SemStamp's setting) next to the hard count it replaces.

Statistic.  For sentences i = 2..n, d_i = depth of sentence i inside the valid set given by the key and sentence i-1's
region (semwm.signed_distance; d > 0 inside).  S = (1 / sqrt(n-1)) * sum_i log p_wm(d_i) / p_null(d_i), with p_wm and
p_null 52-bin histograms of d fitted on one half of the prompts: p_wm on clean Push + Margin texts (reused for every arm,
as in the report), p_null on human texts and the plain model's texts.  Thresholds and rates come from the other half only.
Not specified in the report, chosen here: bins = 50 equal-width bins between the 0.5% and 99.5% quantiles of the pooled
fitting depths + one tail bin on each side; pseudo-count 0.5 per bin; halves by a keyed hash of the prompt id (salt
"lr_split"); texts with fewer than 2 sentences have no score (excluded, as for the hard count).

Thresholds on the evaluation half: human 1% and 0.1% (rank thresholds, posstats.upper_tail_threshold), plain-model 1%,
and the report's "dual null" = the stricter (higher) of human 1% and plain 1%.  The hard count (z_chain, SemStamp's
detector) is re-scored on the same half with the same rules, so the two detectors are compared on identical texts.

Modes:
  depths  (GPU)  per distinct text: sentence depths d_2..d_n and z_chain -> runs/lr_depths/depths.shard<k>.jsonl
  eval    (CPU)  per setting: fit, thresholds, TPR -> runs/lr_detect_<setting>.json, and a table on stdout
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.textnorm import stable_hash  # noqa: E402

ARMS = ("base", "clean_seed1", "semstamp", "chain_reference")
ATTACKS = ("para_light", "para_strong", "pegasus", "pegasus_bigram", "parrot", "parrot_bigram", "parrotss", "parrotss_bigram",
           "reorder_sent", "resegment")
SP, OB = ("sp_realnews", "sp_booksum", "sp_tifu"), ("ob_arxiv", "ob_booksum", "ob_wikipedia")
IT = ("it_dolly", "it_writingprompts", "it_gsm8k", "it_ifeval", "it_translation")
# setting -> (run, null run, domains)
SETTINGS = {f"sp|{m}": (f"sp_bench_{m}", "sp_bench_null", SP) for m in
            ("llama3b", "llama1b", "qwen08b_nl", "llama3b_base", "llama1b_base", "qwen08b_base_nl")}
SETTINGS.update({f"ob|{m}": (r, "ob_bench_null", OB) for m, r in
                 (("llama3b", "ob_bench"), ("llama1b", "ob_bench_llama1b"), ("qwen08b_nl", "ob_bench_qwen08b_nl"),
                  ("llama3b_base", "ob_bench_llama3b_base"), ("llama1b_base", "ob_bench_llama1b_base"), ("qwen08b_base_nl", "ob_bench_qwen08b_base_nl"))})
SETTINGS.update({f"it|{m}": (f"it_bench_{m}", "it_bench_null", IT) for m in ("llama3b", "llama1b", "qwen08b")})
SETTINGS.update({f"setting|{m}": (f"semstamp_setting_{m}", "sp_bench_null", ("sp_realnews",)) for m in ("opt13b_ss", "llama3b_base")})
ROOT = os.environ.get("RUNS_ROOT", "runs")                      # e.g. the /nfs-stor copy of the runs
CACHE = Path(ROOT) / "lr_depths"


def texts_of(run: str, domains, arms=ARMS):
    from wmworld.stages import collect_run_texts
    if not (Path(ROOT) / run).exists():
        return []
    return [g for g in collect_run_texts(Path(ROOT) / run)[0]
            if g.get("status", "ok") == "ok" and g.get("text") and g.get("domain") in domains and g["arm"] in arms]


def all_runs(setting: str) -> list[tuple[str, tuple]]:
    run, null, doms = SETTINGS[setting]
    return [(run, ARMS)] + [(f"{run}_{a}", ARMS) for a in ATTACKS] + [(null, ("base",))]


def run_depths(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--embedder", default=semwm.SEMSTAMP_SBERT if hasattr(semwm, "SEMSTAMP_SBERT") else
                    "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args(argv)
    t0 = time.time()
    texts = {}
    for s in a.settings:
        doms = SETTINGS[s][2]
        for run, arms in all_runs(s):
            for g in texts_of(run, doms, arms):
                texts.setdefault(g["text_hash"], g["text"])
    CACHE.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in glob.glob(str(CACHE / "depths.shard*.jsonl")):
        for ln in open(f):
            try:
                done.add(json.loads(ln)["h"])
            except Exception:
                pass
    hs = sorted(h for h in texts if h not in done and int(stable_hash("lr_shard", h, length=8), 16) % a.num_shards == a.shard)
    print(f"[depths shard {a.shard}] {len(texts)} distinct texts, {len(hs)} to do ({time.time() - t0:.0f}s)", flush=True)
    if not hs:
        return
    emb = semwm.Embedder(a.embedder, device=a.device, batch_size=256)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
    out = open(CACHE / f"depths.shard{a.shard}.jsonl", "a")
    for i in range(0, len(hs), 1024):
        chunk = hs[i:i + 1024]
        sents = [semwm.split_sentences(texts[h]) for h in chunk]
        flat = emb.encode([s for ss in sents for s in ss]) if any(sents) else np.zeros((0, emb.dim))
        k = 0
        for h, ss in zip(chunk, sents):
            P = planes.projections(flat[k:k + len(ss)]) if ss else np.zeros((0, 3)); k += len(ss)
            bins = [planes.bin_of(p) for p in P]
            d = [round(semwm.signed_distance(P[j], semwm.valid_mask(key, bins[j - 1]), planes), 6) for j in range(1, len(bins))]
            out.write(json.dumps({"h": h, "d": d, "z_chain": semwm.chain_z(bins, key)}) + "\n")
        out.flush()
        print(f"  [depths shard {a.shard}] {min(i + 1024, len(hs))}/{len(hs)} ({time.time() - t0:.0f}s)", flush=True)
    out.close()


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def half(prompt_id: str) -> str:
    return "fit" if int(stable_hash("lr_split", prompt_id, length=8), 16) % 2 == 0 else "eval"


class Hist:
    def __init__(self, wm: list[float], null: list[float], nbins: int = 50, pseudo: float = 0.5):
        pool = np.array(wm + null)
        lo, hi = np.quantile(pool, 0.005), np.quantile(pool, 0.995)
        self.edges = np.concatenate([[-np.inf], np.linspace(lo, hi, nbins + 1), [np.inf]])     # 52 bins
        cw = np.histogram(wm, self.edges)[0] + pseudo
        cn = np.histogram(null, self.edges)[0] + pseudo
        self.llr = np.log(cw / cw.sum()) - np.log(cn / cn.sum())

    def score(self, d: list[float]) -> float | None:
        if not d:
            return None
        idx = np.searchsorted(self.edges, np.asarray(d), side="right") - 1
        return float(self.llr[np.clip(idx, 0, len(self.llr) - 1)].sum() / math.sqrt(len(d)))


def run_eval(argv):
    from wmworld.posstats import auroc, upper_tail_threshold, wilson
    from wmworld.records import write_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a = ap.parse_args(argv)
    dep = {}
    for f in glob.glob(str(CACHE / "depths.shard*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln); dep[r["h"]] = r
            except Exception:
                pass
    print(f"{len(dep)} texts with depths", flush=True)
    for s in a.settings:
        run, null, doms = SETTINGS[s]
        groups = defaultdict(list)                                      # (run, arm, domain, half) -> [(h)]
        for r, arms in all_runs(s):
            for g in texts_of(r, doms, arms):
                if g["text_hash"] in dep:
                    groups[(r, g["arm"], g["domain"], half(g["prompt_id"]))].append(g["text_hash"])
        fit_wm = [x for d in doms for h in groups[(run, "chain_reference", d, "fit")] for x in dep[h]["d"]]
        fit_null = [x for d in doms for rr, arm in ((null, "base"), (run, "base")) for h in groups[(rr, arm, d, "fit")] for x in dep[h]["d"]]
        if not fit_wm or not fit_null:
            print(f"{s}: no fitting data, skipped"); continue
        H = Hist(fit_wm, fit_null)
        val = {"lr": lambda h: H.score(dep[h]["d"]), "hard": lambda h: dep[h]["z_chain"]}
        rep = {"setting": s, "run": run, "null": null, "n_fit_wm_sentences": len(fit_wm), "n_fit_null_sentences": len(fit_null),
               "edges": [float(x) for x in H.edges[1:-1]], "llr": [float(x) for x in H.llr], "thresholds": {}, "results": {}}
        for d in doms:
            for st, fn in val.items():
                hv = [x for x in (fn(h) for h in groups[(null, "base", d, "eval")]) if x is not None]
                pv = [x for x in (fn(h) for h in groups[(run, "base", d, "eval")]) if x is not None]
                t = {"human_1%": upper_tail_threshold(hv, 0.01) if hv else None, "human_0.1%": upper_tail_threshold(hv, 0.001) if hv else None,
                     "plain_1%": upper_tail_threshold(pv, 0.01) if pv else None, "n_human": len(hv), "n_plain": len(pv)}
                cands = [x for x in (t["human_1%"], t["plain_1%"]) if x is not None and np.isfinite(x)]
                t["dual_1%"] = max(cands) if cands else None
                rep["thresholds"][f"{d}:{st}"] = t
                rep["thresholds"][f"{d}:{st}"]["_hv"] = hv
        for (r, arm, d, hf), hs in sorted(groups.items()):
            if hf != "eval" or r == null:
                continue
            rec = {"n": len(hs)}
            for st, fn in val.items():
                v = [x for x in (fn(h) for h in hs) if x is not None]
                th = rep["thresholds"][f"{d}:{st}"]
                for tn in ("dual_1%", "human_1%", "human_0.1%"):
                    thr = th[tn]
                    if thr is None or not np.isfinite(thr) or not v:
                        rec[f"{st}:{tn}"] = None; continue
                    k = sum(x > thr for x in v); lo, hi = wilson(k, len(v))
                    rec[f"{st}:{tn}"] = {"rate": k / len(v), "ci": [lo, hi]}
                rec[f"{st}:auroc_vs_human"] = auroc(v, th["_hv"]) if (v and th["_hv"]) else None
            rep["results"][f"{r}|{arm}|{d}"] = rec
        for k in rep["thresholds"]:
            rep["thresholds"][k].pop("_hv")
        out = f"{ROOT}/lr_detect_{s.replace('|', '_')}.json"
        write_json(out, rep)
        f = lambda x: "  -  " if not x else f"{100 * x['rate']:5.1f}"
        print(f"\n== {s} (fit: {len(fit_wm)} push sentences, {len(fit_null)} null sentences) -> {out}")
        print(f"{'run':34s} {'arm':16s} {'domain':18s} {'n':>4s} | hard dual1% h1% h0.1% | LR dual1% h1% h0.1%")
        for key_, rec in sorted(rep["results"].items()):
            r, arm, d = key_.split("|")
            if arm == "base" and r != run:
                continue
            print(f"{r:34s} {arm:16s} {d:18s} {rec['n']:4d} | {f(rec['hard:dual_1%'])} {f(rec['hard:human_1%'])} {f(rec['hard:human_0.1%'])} | "
                  f"{f(rec['lr:dual_1%'])} {f(rec['lr:human_1%'])} {f(rec['lr:human_0.1%'])}", flush=True)


def main():
    mode, argv = sys.argv[1], sys.argv[2:]
    {"depths": run_depths, "eval": run_eval}.get(mode, lambda _: sys.exit(f"unknown mode {mode}"))(argv)


if __name__ == "__main__":
    main()
