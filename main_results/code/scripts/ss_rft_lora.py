"""Rejection-sampling fine-tuning of the mask adapters (amendment 16): the warm start before RL.

Policy exactly as in scripts/ss_rl_mask.py (frozen Llama-3.2-3B + 8 LoRA adapters r 16, adapter index = mask id =
previous sentence's bin; adapter on from the last prefix token), so the result can start an RL run.
  round 1   accepted sentences of the training corpus (runs/ss_corpus: 16 plain-3B samples per sentence position):
            accepted = the chain's acceptance, in the mask AND >= 0.02 from its planes
  round r>1 on 1,500 fresh training contexts, 16 sentences per context from the CURRENT adapters; accepted = same rule
            and no guard failure (proper ending, no repetition)
Each round: 1 epoch of next-token likelihood (tempered logits, T 0.8) on the accepted sentences, continuing from the
previous round's adapters; lr 5e-5, gradient accumulation over 4 contexts.  After each round: the RL monitor (same 48
validation contexts, 16 sentences each, same seeds, local Qwen3.5-0.8B judge) and a checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ss_rl_mask", ROOT / "scripts" / "ss_rl_mask.py")
rl = importlib.util.module_from_spec(spec); sys.modules["ss_rl_mask"] = rl; spec.loader.exec_module(rl)
ss = rl.ss

from wmworld import multilora, semwm  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import seed_from  # noqa: E402


def corpus_accepted(run: str, contexts: list[dict], accept: str = "clear") -> list[dict]:
    """Round 1: per training context, the logged candidates that the chain would accept (accept="clear": in the mask
    AND >= 0.02 from its planes) or every logged candidate (accept="all": distillation of the sampler that wrote the
    corpus, e.g. the push -- MLE on its samples minimises KL(sampler || adapter))."""
    corpus = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); corpus[r["prompt_id"]] = r
    planes = semwm.Hyperplanes(np.eye(3))                                       # only signs of logged projections are used
    out = []
    for it in contexts:
        u = next(u for u in corpus[it["pid"]]["units"] if u["unit"] == it["unit"])
        acc = [c["ids"] for c in u["cands"] if c["ids"] and (accept == "all" or semwm.signed_distance(np.array(c["proj"]), it["mask"], planes) >= semwm.MARGIN)]
        if acc:
            out.append(it | {"accepted": acc})
    return out


def sft_round(lm, ctrl, params, items, a, t0, tag) -> dict:
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    random.Random(seed_from("rft", tag)).shuffle(items)
    tot, ntok = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for k, it in enumerate(items, 1):
        cands = it["accepted"][: a.max_per_context]
        lp, msk = rl.token_logprobs(lm.model, ctrl, it["prefix"], cands, [it["prev_bin"]] * len(cands), a.temperature, lm.pad_id, lm.device, True)
        mf = msk.float()
        loss = -(lp * mf).sum() / mf.sum()
        (loss / a.accum).backward()
        tot += float(loss.detach()) * float(mf.sum()); ntok += int(mf.sum())
        if k % a.accum == 0 or k == len(items):
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if k % 500 == 0:
            print(f"  [{tag}] {k}/{len(items)} contexts, nll/token {tot / max(1, ntok):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return {"contexts": len(items), "sentences": sum(min(len(it["accepted"]), a.max_per_context) for it in items), "nll_per_token": tot / max(1, ntok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/ss_corpus")
    ap.add_argument("--val", default="runs/ss_corpus_val")
    ap.add_argument("--out", default="runs/ss_rft/v1")
    ap.add_argument("--ckpt-dir", default="/l/users/daniil.orel/wmworld_rl/rft_v1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-contexts", type=int, default=1500)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-per-context", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--accept", choices=["clear", "all"], default="clear", help="round 1: accepted candidates only, or all (distillation)")
    a = ap.parse_args()
    a.w_depth, a.w_judge = 1.0, 1.0                                             # the RL reward, used only for monitoring
    t0 = time.time()
    random.seed(0); torch.manual_seed(0)
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    lm = ss.LM(m["path"], "bf16", device="cuda:0")
    ctrl = multilora.attach(lm.model, n=8, r=16, alpha=32.0)
    params = multilora.trainable(lm.model)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=64)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    judge = rl.Judge("cuda:0")
    sampling = SimpleNamespace(temperature=a.temperature, top_p=0.95)
    unit_a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, W=16)
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    train, val = rl.load_contexts(a.train, prompts, lm.tokenizer), rl.load_contexts(a.val, prompts, lm.tokenizer)
    val_mon = random.Random(1).sample(val, min(a.eval_n, len(val)))                 # the RL runs' monitor contexts
    ckpt = Path(a.ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(a), "rounds": [], "monitor": []}
    pr = lambda mo: (f"in-mask {mo['in_mask']:.3f} clear {mo['clear_0.02']:.3f} depth {mo['depth']:+.4f} judge {mo['judge']:.3f} "
                     f"guard {mo['guard_rate']:.3f} kl/sent {mo['kl_per_sentence']:.3f}")
    base = rl.monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, a, lm.pad_id, False, 0)
    log["monitor"].append(base); print(f"[monitor base] {pr(base)}", flush=True)
    used = set()
    for rnd in range(1, a.rounds + 1):
        if rnd == 1:
            items = corpus_accepted(a.train, train, a.accept)
        else:
            pool = [i for i in range(len(train)) if i not in used]
            pick = random.Random(seed_from("rft_round", rnd)).sample(pool, min(a.round_contexts, len(pool)))
            used.update(pick)
            gen = torch.Generator(device=lm.device)
            items, n_samp, n_acc = [], 0, 0
            for j, i in enumerate(pick, 1):
                it = train[i]
                gen.manual_seed(seed_from("rft_sample", rnd, i))
                with torch.no_grad():
                    ro = rl.rollout(lm, ctrl, emb, planes, judge, it, a.group, sampling, unit_a, gen, a, True)
                n_samp += len(ro)
                acc = [x["ids"] for x in ro if x["clear"] and x["guard"] == 0]
                n_acc += len(acc)
                if acc:
                    items.append(it | {"accepted": acc})
                if j % 250 == 0:
                    print(f"  [round {rnd} sampling] {j}/{len(pick)} contexts, accepted {n_acc}/{n_samp} = {n_acc / max(1, n_samp):.3f} ({time.time() - t0:.0f}s)", flush=True)
            print(f"[round {rnd}] on-policy acceptance on training contexts: {n_acc}/{n_samp} = {n_acc / max(1, n_samp):.3f}", flush=True)
        stats = sft_round(lm, ctrl, params, items, a, t0, f"round {rnd}")
        mo = rl.monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, a, lm.pad_id, True, rnd)
        log["rounds"].append(stats | {"round": rnd}); log["monitor"].append(mo)
        print(f"[round {rnd}] trained on {stats['sentences']} accepted sentences from {stats['contexts']} contexts (nll/token {stats['nll_per_token']:.4f}) | "
              f"[monitor] {pr(mo)} ({time.time() - t0:.0f}s)", flush=True)
        multilora.save(ckpt / f"adapters_round{rnd}.pt", lm.model, {"key": semwm.Key.shipped().label(), "round": rnd,
                                                                    "adapter_index": "prev_bin (mask id)", "monitor": {k: v for k, v in mo.items() if k != "samples"}})
        write_json(Path(a.out) / "rft_log.json", log)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
