"""Train the bin adapters (amendment 11): 8 LoRA adapters on the frozen Llama-3.2-3B, one per LSH bin of our key.

Data = the candidate corpus (scripts/steered_semstamp.py --arm chain --best-of 16 --log-cands): at every sentence
position, 16 exact samples of the 3B's own sampler, each labelled with the bin of its SemStamp embedding.  Each
candidate is a maximum-likelihood example for ITS bin's adapter, so adapter b is fitted to the model's own sampler
restricted to bin b.

Adapter scope (same at inference): the prefix up to its LAST token is encoded by the base model; the adapter is on
from the last prefix token onward (so it also shapes the first token of the sentence).  One batch = one context: the
prefix minus its last token is run once without gradients, its cache repeated for the candidates, and the candidates
(input = last prefix token + candidate[:-1]) are run with per-row adapter = candidate bin.
The loss is on logits / T (T = the sampler's temperature, 0.8): the data are tempered samples, and at inference the
adapter's logits go through the same temperature + top-k + top-p as the base, so at initialisation adapter == base.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import multilora, semwm  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM  # noqa: E402
from wmworld.records import content_hash, write_json  # noqa: E402

DATE = "26 Jul 2024"


def load_contexts(runs: list[str], prompts: dict[str, str], tok, min_cands: int = 2) -> list[dict]:
    """One item per (text, unit >= 1): prefix ids, candidate ids, candidate bins.  Last corpus line per prompt wins."""
    corpus = {}
    for run in runs:
        for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
            for ln in open(f):
                r = json.loads(ln)
                corpus[r["prompt_id"]] = r
    items = []
    for pid, c in sorted(corpus.items()):
        chat = tok.apply_chat_template([{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": prompts[pid]}],
                                       tokenize=False, add_generation_prompt=True, date_string=DATE)
        prefix = [int(t) for t in tok(chat, add_special_tokens=False)["input_ids"]]
        for u in c["units"]:
            if u["mask"] is not None and len(u["cands"]) >= min_cands:
                items.append({"pid": pid, "unit": u["unit"], "prefix": list(prefix),
                              "cands": [cd["ids"] for cd in u["cands"] if cd["ids"]], "bins": [cd["bin"] for cd in u["cands"] if cd["ids"]]})
            prefix += u["cands"][u["chosen"]]["ids"]
    return items


def batch_nll(model, ctrl, item: dict, rows_adapter: list[int], T: float, pad_id: int, device, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed token NLL per candidate (tempered logits) and token counts, with adapter index per row (-1 = base)."""
    pre, last = item["prefix"][:-1], item["prefix"][-1]
    R = len(item["cands"])
    ctrl.set_rows(None)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([pre], device=device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(R)
    Lc = max(len(c) for c in item["cands"])
    inp = torch.full((R, Lc), pad_id, dtype=torch.long, device=device)
    tgt = torch.full((R, Lc), -100, dtype=torch.long, device=device)
    att = torch.zeros((R, len(pre) + Lc), dtype=torch.long, device=device)
    att[:, :len(pre)] = 1
    for r, c in enumerate(item["cands"]):
        seq = [last] + c[:-1]
        inp[r, :len(seq)] = torch.tensor(seq, device=device)
        tgt[r, :len(c)] = torch.tensor(c, device=device)
        att[r, len(pre):len(pre) + len(seq)] = 1
    ctrl.set_rows(rows_adapter)
    with torch.set_grad_enabled(grad):
        logits = model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).logits
        lp = F.log_softmax(logits.float() / T, dim=-1)
        nll = -lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * (tgt >= 0)
    ctrl.set_rows(None)
    return nll.sum(dim=1), (tgt >= 0).sum(dim=1)


@torch.no_grad()
def evaluate(model, ctrl, items, T, pad_id, device) -> dict:
    """Per-token NLL of held-out candidates: own-bin adapter, base (adapters off), and a different bin's adapter."""
    tot = {"own": 0.0, "base": 0.0, "other": 0.0}
    n = 0
    rng = random.Random(0)
    for it in items:
        own = it["bins"]
        other = [(b + rng.randrange(1, 8)) % 8 for b in own]
        s_own, cnt = batch_nll(model, ctrl, it, own, T, pad_id, device, False)
        s_base, _ = batch_nll(model, ctrl, it, [-1] * len(own), T, pad_id, device, False)
        s_oth, _ = batch_nll(model, ctrl, it, other, T, pad_id, device, False)
        tot["own"] += float(s_own.sum()); tot["base"] += float(s_base.sum()); tot["other"] += float(s_oth.sum()); n += int(cnt.sum())
    return {k: v / max(1, n) for k, v in tot.items()} | {"tokens": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", default=[], help="corpus run dir (train prompts)")
    ap.add_argument("--val", action="append", default=[], help="corpus run dir (validation prompts)")
    ap.add_argument("--models-json", default="assets/ss_models.json")
    ap.add_argument("--out", default="runs/ss_adapters/v1")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=1000, help="contexts")
    ap.add_argument("--eval-n", type=int, default=300, help="validation contexts per evaluation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed); random.seed(a.seed)
    m = json.loads(Path(a.models_json).read_text())["llama3b"]
    tok = AutoTokenizer.from_pretrained(m["path"])
    pad_id = int(tok.convert_tokens_to_ids("<|finetune_right_pad_id|>"))
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    train, val = load_contexts(a.train, prompts, tok), load_contexts(a.val, prompts, tok)
    rnd = random.Random(1)
    val_eval = rnd.sample(val, min(a.eval_n, len(val)))
    counts = Counter(b for it in train for b in it["bins"])
    print(f"train contexts {len(train)} ({sum(len(it['cands']) for it in train)} candidates), val contexts {len(val)} "
          f"(eval subset {len(val_eval)}); train bin counts " + " ".join(f"{b}:{counts[b]}" for b in range(8)), flush=True)

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(m["path"], dtype=torch.bfloat16).to(device).eval()
    ctrl = multilora.attach(model, n=8, r=a.r, alpha=a.alpha)
    params = multilora.trainable(model)
    print(f"adapters: {model._multilora['n_modules']} modules, {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters "
          f"({time.time() - t0:.0f}s)", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = int(math.ceil(len(train) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    meta = {"key": semwm.Key.shipped().label(), "n_bins": 8, "train_runs": a.train, "val_runs": a.val, "temperature": a.temperature,
            "corpus_hash": content_hash({"train": sorted({it["pid"] for it in train}), "n": len(train)}, length=12), "args": vars(a)}
    log = {"meta": meta, "bin_counts": dict(counts), "evals": []}
    ev = evaluate(model, ctrl, val_eval, a.temperature, pad_id, device)
    ev.update({"contexts": 0, "step": 0, "wall": time.time() - t0}); log["evals"].append(ev)
    print(f"[eval 0] own {ev['own']:.4f} base {ev['base']:.4f} other {ev['other']:.4f} (per token, {ev['tokens']} tokens)", flush=True)
    best = ev["own"]
    multilora.save(Path(a.out) / "adapters.pt", model, meta | {"best_eval": ev})
    order = [i for _ in range(int(math.ceil(a.epochs))) for i in rnd.sample(range(len(train)), len(train))][: int(len(train) * a.epochs)]
    step, run_loss, run_tok = 0, 0.0, 0
    model.train()
    for n_ctx, i in enumerate(order, 1):
        it = train[i]
        s, cnt = batch_nll(model, ctrl, it, it["bins"], a.temperature, pad_id, device, True)
        (s.sum() / max(1, int(cnt.sum())) / a.accum).backward()
        run_loss += float(s.detach().sum()); run_tok += int(cnt.sum())
        if n_ctx % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if n_ctx % 200 == 0:
            print(f"[train] contexts {n_ctx}/{len(order)} step {step} loss/token {run_loss / max(1, run_tok):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run_loss, run_tok = 0.0, 0
        if n_ctx % a.eval_every == 0 or n_ctx == len(order):
            model.eval()
            ev = evaluate(model, ctrl, val_eval, a.temperature, pad_id, device)
            ev.update({"contexts": n_ctx, "step": step, "wall": time.time() - t0}); log["evals"].append(ev)
            print(f"[eval {n_ctx}] own {ev['own']:.4f} base {ev['base']:.4f} other {ev['other']:.4f}", flush=True)
            if ev["own"] < best:
                best = ev["own"]
                multilora.save(Path(a.out) / "adapters.pt", model, meta | {"best_eval": ev})
                print(f"  saved best ({best:.4f})", flush=True)
            write_json(Path(a.out) / "train_log.json", log)
            model.train()
    write_json(Path(a.out) / "train_log.json", log)
    print(f"done ({time.time() - t0:.0f}s); best own-adapter val NLL {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
