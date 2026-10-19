"""Validation measurements of the bin adapters (amendment 11, step 6), on VALIDATION contexts only (runs/ss_corpus_val).

For each sampled context (a unit >= 1 of a validation text, with its mask) and each of the mask's 2 bins b:
  - n samples from adapter b (scripts/steered_semstamp.py adapter_candidates), embedded;
  - share of them landing in b, vs the base's share at the same context (the 16 logged base candidates);
  - share in the mask AND clearing 0.02 (per candidate), adapters (both mask bins pooled) vs base;
  - per-token NLL under the base's processed sampler of adapter samples vs of the logged base samples;
  - log q_b(x) - log p(x) per sentence for adapter samples (a per-sentence KL(q_b||p) estimate, nats);
  - SIR: effective sample size of w = p / q_mix over the in-region adapter candidates of the context.
Numbers only; written to <adapters dir>/val_eval.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("steered_semstamp", ROOT / "scripts" / "steered_semstamp.py")
ss = importlib.util.module_from_spec(spec); sys.modules["steered_semstamp"] = ss; spec.loader.exec_module(ss)
tr_spec = importlib.util.spec_from_file_location("ss_train_adapters", ROOT / "scripts" / "ss_train_adapters.py")
tr = importlib.util.module_from_spec(tr_spec); sys.modules["ss_train_adapters"] = tr; tr_spec.loader.exec_module(tr)

from wmworld import multilora, semwm  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import seed_from  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="runs/ss_adapters/v1/adapters.pt")
    ap.add_argument("--val", default="runs/ss_corpus_val")
    ap.add_argument("--contexts", type=int, default=200)
    ap.add_argument("--n", type=int, default=8, help="adapter samples per mask bin per context")
    a0 = ap.parse_args()
    t0 = time.time()
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    lm = ss.LM(m["path"], "bf16")
    ctrl, meta = multilora.load(a0.adapters, lm.model)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device=str(lm.device), batch_size=64)
    key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
    sampling = SimpleNamespace(temperature=0.8, top_p=0.95)
    a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, W=16)
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    items = tr.load_contexts([a0.val], prompts, lm.tokenizer)
    rng = np.random.default_rng(0)
    pick = [items[i] for i in rng.choice(len(items), size=min(a0.contexts, len(items)), replace=False)]
    # the corpus stores the mask per unit: rebuild it from the logged bins of the chosen previous unit
    corpus = {}
    import glob
    for f in sorted(glob.glob(f"{a0.val}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); corpus[r["prompt_id"]] = r
    gen = torch.Generator(device=lm.device)
    res = {"base_in_b": [], "adapt_in_b": [], "base_region": [], "adapt_region": [], "nll_base_samples": [], "nll_adapt_samples": [],
           "logq_minus_logp": [], "sir_ess": [], "sir_n": [], "per_bin": {str(b): {"n": 0, "hit": 0} for b in range(8)}}
    for c_i, it in enumerate(pick):
        unit = next(u for u in corpus[it["pid"]]["units"] if u["unit"] == it["unit"])
        mask = tuple(unit["mask"])
        base_bins = [cd["bin"] for cd in unit["cands"]]
        base_P = np.array([cd["proj"] for cd in unit["cands"]])
        res["base_region"].append(float(np.mean([semwm.signed_distance(q, mask, planes) > 0 and np.min(np.abs(q)) >= semwm.MARGIN for q in base_P])))
        # base samples' per-token NLL under the base sampler
        Lb = ss.unit_logprobs(lm, ctrl, it["prefix"], it["cands"], [-1], sampling)
        res["nll_base_samples"].append(float(-Lb[:, 0].sum() / sum(len(c) for c in it["cands"])))
        pooled = []
        for b in mask:
            gen.manual_seed(seed_from("adapter_eval", it["pid"], it["unit"], b))
            cands, _ = ss.adapter_candidates(lm, ctrl, it["prefix"], [int(b)] * a0.n, sampling, a, 64, 0, 10 ** 6, gen)
            if not cands:
                continue
            P = planes.projections(emb.encode([c["text"] for c in cands]))
            hits = [planes.bin_of(q) == b for q in P]
            res["adapt_in_b"].append(float(np.mean(hits))); res["base_in_b"].append(float(np.mean([x == b for x in base_bins])))
            res["per_bin"][str(b)]["n"] += len(hits); res["per_bin"][str(b)]["hit"] += int(sum(hits))
            L = ss.unit_logprobs(lm, ctrl, it["prefix"], [c["ids"] for c in cands], [-1, int(b)] + [int(x) for x in mask if x != b], sampling)
            res["nll_adapt_samples"].append(float(-L[:, 0].sum() / sum(len(c["ids"]) for c in cands)))
            res["logq_minus_logp"] += [float(x) for x in (L[:, 1] - L[:, 0])]
            for c, q, row in zip(cands, P, L):
                pooled.append((c, q, row[0], np.logaddexp(row[1], row[2]) - np.log(2)))
        region = [(c, q, lp, lq) for c, q, lp, lq in pooled if semwm.signed_distance(q, mask, planes) > 0 and np.min(np.abs(q)) >= semwm.MARGIN]
        res["adapt_region"].append(len(region) / max(1, len(pooled)))
        if region:
            lw = np.array([lp - lq for _, _, lp, lq in region]); lw = lw[np.isfinite(lw)]
            if len(lw):
                w = np.exp(lw - lw.max()); res["sir_ess"].append(float(w.sum() ** 2 / (w ** 2).sum())); res["sir_n"].append(len(lw))
        if (c_i + 1) % 20 == 0:
            print(f"[{c_i + 1}/{len(pick)}] in-bin adapter {np.mean(res['adapt_in_b']):.3f} vs base {np.mean(res['base_in_b']):.3f} | "
                  f"region adapter {np.mean(res['adapt_region']):.3f} vs base {np.mean(res['base_region']):.3f} ({time.time() - t0:.0f}s)", flush=True)
    summ = {"contexts": len(pick), "n_per_bin": a0.n, "adapters": a0.adapters, "meta_best_eval": meta.get("best_eval"),
            "share_in_target_bin": {"adapter": float(np.mean(res["adapt_in_b"])), "base_same_contexts": float(np.mean(res["base_in_b"]))},
            "share_in_mask_and_clear_0.02": {"adapter": float(np.mean(res["adapt_region"])), "base_same_contexts": float(np.mean(res["base_region"]))},
            "nll_per_token_under_base": {"adapter_samples": float(np.mean(res["nll_adapt_samples"])), "base_samples": float(np.mean(res["nll_base_samples"]))},
            "logq_minus_logp_per_sentence": {"mean": float(np.mean(res["logq_minus_logp"])), "median": float(np.median(res["logq_minus_logp"]))},
            "sir_ess": {"mean": float(np.mean(res["sir_ess"])) if res["sir_ess"] else None, "mean_n": float(np.mean(res["sir_n"])) if res["sir_n"] else None,
                        "contexts_with_region_candidates": len(res["sir_ess"])},
            "per_bin_hit_rate": {b: (v["hit"] / v["n"] if v["n"] else None) for b, v in res["per_bin"].items()}}
    out = Path(a0.adapters).with_name("val_eval.json")
    write_json(out, summ)
    print(json.dumps(summ, indent=1)); print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
