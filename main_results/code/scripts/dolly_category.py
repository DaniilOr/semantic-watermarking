"""Dolly results split by category (open_qa, general_qa, brainstorming, creative_writing): detection (SemStamp's z-score,
threshold = 99th / 99.9th percentile of the human Dolly answers of runs/it_bench_null, all categories pooled, as in the
main results) and the pairwise judge (api_pref_*.jsonl rows of the instruction runs).  Read-only.
Usage: python scripts/dolly_category.py --out runs/dolly_category.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from api_judge import summarize_preference  # noqa: E402
from wmworld import semwm  # noqa: E402
from wmworld.stages import collect_run_texts  # noqa: E402

ENC = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"
MODELS = ("llama3b", "llama1b", "qwen08b")
ARMS = ("semstamp", "chain_reference")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    cat = {r["prompt_id"]: r.get("category") for r in map(json.loads, open("data/it_bench/prompts.jsonl", encoding="utf-8"))
           if r["domain"] == "it_dolly"}
    emb = semwm.Embedder(ENC, device="cpu", batch_size=256)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)

    def z(t):
        return semwm.score_text(t, key, emb, W=16, planes=planes)["z_chain"]

    null = [r["text"] for r in collect_run_texts(Path("runs/it_bench_null"))[0] if r.get("domain") == "it_dolly" and r.get("text")]
    hz = np.array([v for v in map(z, null) if v is not None])
    t1, t01 = float(np.quantile(hz, 0.99)), float(np.quantile(hz, 0.999))
    print(f"human Dolly n={len(hz)} thresholds 1% {t1:.2f} 0.1% {t01:.2f}", flush=True)
    out = {"thresholds": {"human_1%": t1, "human_0.1%": t01, "n_human": int(len(hz))}, "results": {}}
    for m in MODELS:
        run = f"runs/it_bench_{m}"
        recs = [r for r in collect_run_texts(Path(run))[0] if r.get("domain") == "it_dolly" and r.get("arm") in ARMS
                and r.get("status", "ok") == "ok" and r.get("text")]
        recs = list({(r["arm"], r["prompt_id"]): r for r in recs}.values())
        jl = Path(run) / "eval/api_pref_google_gemma-4-31b-it.jsonl"
        jrows = [r for r in map(json.loads, open(jl, encoding="utf-8")) if r.get("domain") == "it_dolly"] if jl.exists() else []
        for arm in ARMS + ("clean_seed1",):
            by = defaultdict(list)
            for r in recs:
                if r["arm"] == arm:
                    v = z(r["text"])
                    if v is not None:
                        by[cat.get(r["prompt_id"])].append(v)
            jby = defaultdict(list)
            for r in jrows:
                if r.get("arm") == arm:
                    jby[cat.get(r["prompt_id"])].append(r)
            for c in sorted(set(by) | set(jby), key=str):
                zs = np.array(by.get(c, []))
                rec = {"n": int(len(zs))}
                if len(zs):
                    rec.update({"tpr_1%": float((zs > t1).mean()), "tpr_0.1%": float((zs > t01).mean())})
                if jby.get(c):
                    s = summarize_preference(jby[c]); rec.update({"judge_win": s["win_rate"], "judge_n": s["n_prompts"]})
                out["results"][f"{m}|{arm}|{c}"] = rec
                print(f"{m:8s} {arm:16s} {str(c):17s} {json.dumps(rec)}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
