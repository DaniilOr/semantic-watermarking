"""Token-level distillation of the push into the mask-indexed LoRA adapters (amendment 20).

The sequence-level distillation (fine-tuning on pushed samples) left the adapters at plain-model level although the
pushed corpus was clearly in the mask more often (0.324 vs 0.242 per candidate).  Here every token gets the push's full
target distribution instead of one sampled token.

  teacher  For each training context of runs/ss_corpus (one random logged plain candidate per sentence position), one
           base-model pass gives the next-token logits at every position of that sentence; the push
           (semwm.NucleusSteerProcessor, true-shift / marginal, beta 1, the settings of the push arms, target = the
           position's mask) turns them into the pushed logits over the model's top-p nucleus.  Stored per position:
           nucleus token ids and pushed logits.  Also measured: KL(pushed || base) over the nucleus, per sentence.
  train    Adapters (8, r 16, index = previous sentence's bin; prefix[:-1] by the base model, adapter from the last
           prefix token -- the RL scope) minimise KL(pushed || student) over the same nucleus at every position,
           T 0.8.  RL monitor (local judge, 48 contexts x 16) before and after; checkpoint per epoch.
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
from wmworld.textnorm import seed_from  # noqa: E402

T, TOP_P = 0.8, 0.95


def contexts_with_candidate(run: str, lm) -> list[dict]:
    """rl.load_contexts items (prefix, gen_start, mask, prev_bin) + one random logged candidate of that position."""
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    items = rl.load_contexts(run, prompts, lm.tokenizer)
    corpus = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); corpus[r["prompt_id"]] = r
    out = []
    for it in items:
        u = next(u for u in corpus[it["pid"]]["units"] if u["unit"] == it["unit"])
        cands = [c["ids"] for c in u["cands"] if c["ids"]]
        if cands:
            out.append(it | {"cand": cands[random.Random(seed_from("tokdist", it["pid"], it["unit"])).randrange(len(cands))]})
    return out


@torch.no_grad()
def teacher(a) -> None:
    lm = ss.LM(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"], "bf16", device="cuda:0")
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=64)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    eos = lm.eos_ids + [lm.pad_id]
    boundary = semwm.boundary_token_mask(lm.tokenizer, eos, lm.device)
    items = contexts_with_candidate(a.corpus, lm)
    if a.limit:
        items = items[: a.limit]
    mine = [it for i, it in enumerate(items) if i % a.num_shards == a.shard]
    eos_t = torch.tensor(sorted(set(lm.eos_ids)), device=lm.device)
    recs, kls, t0 = [], [], time.time()
    for n, it in enumerate(mine, 1):
        pre, ids = it["prefix"], it["cand"]
        seq = torch.tensor([pre + ids[:-1]], device=lm.device)
        logits = lm.model(input_ids=seq).logits[0, len(pre) - 1:].float()              # [L, V]: row j predicts ids[j]
        proc = semwm.NucleusSteerProcessor(planes, emb, lm.tokenizer, len(pre), it["mask"][0], 1.0, boundary, T, TOP_P,
                                           mode="marginal", stat="seg", topk=20, clip_sd=1.0, target_depth=0.05, rep_n=4,
                                           rep_L0=it["gen_start"], seg_W=None, eos_ids=eos, target_bins=tuple(it["mask"]))
        nucs, tlogs, kl = [], [], 0.0
        for j in range(len(ids)):
            s = logits[j].clone()
            if j < 2:
                s[eos_t] = float("-inf")                                                 # min_new_tokens = 2, as in sampling
            new = proc(torch.tensor([pre + ids[:j]], device=lm.device), s[None].clone())[0]
            nuc = proc._nucleus(s)                      # the sampler's top-p set; where the push acts, new is -inf outside it,
            if not torch.isfinite(new[nuc]).all():      # where it does not act (confident step), new == s over the full vocab
                nuc = torch.nonzero(torch.isfinite(new)).flatten()
            pt = torch.log_softmax(new[nuc] / T, -1); pb = torch.log_softmax(s[nuc] / T, -1)
            kl += float((pt.exp() * (pt - pb)).sum())
            nucs.append(nuc.cpu().int()); tlogs.append(new[nuc].cpu().half())
        kls.append(kl)
        recs.append({"pid": it["pid"], "unit": it["unit"], "prev_bin": it["prev_bin"], "prefix": pre, "ids": ids, "nuc": nucs, "tlog": tlogs})
        if n % 200 == 0 or n == len(mine):
            print(f"[teacher shard {a.shard}] {n}/{len(mine)} sentences, KL(push||base) per sentence mean {np.mean(kls):.3f} "
                  f"median {np.median(kls):.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"recs": recs, "kl_push_base": kls}, out / f"teacher.shard{a.shard}.pt")
    print(f"wrote {out}/teacher.shard{a.shard}.pt ({len(recs)} sentences)", flush=True)


def student_logits(model, ctrl, prefix: list[int], ids: list[int], row: int, device) -> torch.Tensor:
    """[L, V] logits for ids under adapter `row`; prefix[:-1] encoded by the base model, adapter from the last prefix token."""
    ctrl.set_rows(None)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix[:-1]], device=device), use_cache=True).past_key_values
    ctrl.set_rows([row])
    try:
        out = model(input_ids=torch.tensor([[prefix[-1]] + ids[:-1]], device=device), past_key_values=cache, use_cache=True).logits[0]
    finally:
        ctrl.set_rows(None)
    return out.float()


def train(a) -> None:
    random.seed(0); torch.manual_seed(0)
    lm = ss.LM(json.loads(Path("assets/ss_models.json").read_text())["llama3b"]["path"], "bf16", device="cuda:0")
    ctrl = multilora.attach(lm.model, n=8, r=16, alpha=32.0)
    params = multilora.trainable(lm.model)
    for q in params:
        q.requires_grad_(True)
    recs, kl_pb = [], []
    for f in sorted(glob.glob(f"{a.teacher}/teacher.shard*.pt")):
        blob = torch.load(f, weights_only=False); recs += blob["recs"]; kl_pb += blob["kl_push_base"]
    print(f"teacher: {len(recs)} sentences, {sum(len(r['ids']) for r in recs)} positions; KL(push||base) per sentence "
          f"mean {np.mean(kl_pb):.3f} median {np.median(kl_pb):.3f}", flush=True)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=64)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    judge = rl.Judge("cuda:0")
    sampling = SimpleNamespace(temperature=T, top_p=TOP_P)
    unit_a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, W=16)
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    val_mon = random.Random(1).sample(rl.load_contexts(a.val, prompts, lm.tokenizer), a.eval_n)
    ma = SimpleNamespace(w_depth=1.0, w_judge=1.0, temperature=T)
    pr = lambda mo: (f"in-mask {mo['in_mask']:.3f} clear {mo['clear_0.02']:.3f} depth {mo['depth']:+.4f} judge {mo['judge']:.3f} "
                     f"guard {mo['guard_rate']:.3f} kl/sent {mo['kl_per_sentence']:.3f}")
    log = {"args": vars(a), "teacher_kl_push_base": {"mean": float(np.mean(kl_pb)), "median": float(np.median(kl_pb))}, "monitor": [], "train": []}
    base = rl.monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, ma, lm.pad_id, False, 0)
    log["monitor"].append(base); print(f"[monitor base] {pr(base)}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    ck = Path(a.ckpt_dir); ck.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        order = random.Random(seed_from("tokdist_ep", ep)).sample(range(len(recs)), len(recs))
        tot, npos = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, i in enumerate(order, 1):
            r = recs[i]
            lg = student_logits(lm.model, ctrl, r["prefix"], r["ids"], int(r["prev_bin"]), lm.device)
            loss = 0.0
            for j, (nuc, tl) in enumerate(zip(r["nuc"], r["tlog"])):
                nuc = nuc.to(lm.device).long()
                pt = torch.log_softmax(tl.to(lm.device).float() / T, -1)
                qs = torch.log_softmax(lg[j, nuc] / T, -1)
                loss = loss + (pt.exp() * (pt - qs)).sum()
            loss = loss / len(r["ids"])
            (loss / a.accum).backward()
            tot += float(loss.detach()) * len(r["ids"]); npos += len(r["ids"])
            if k % a.accum == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if k % 1000 == 0:
                print(f"  [epoch {ep}] {k}/{len(order)} sentences, KL(push||student) per token {tot / npos:.4f} ({time.time() - t0:.0f}s)", flush=True)
                log["train"].append({"epoch": ep, "k": k, "kl_per_token": tot / npos})
        mo = rl.monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, ma, lm.pad_id, True, ep)
        log["monitor"].append(mo); print(f"[monitor epoch {ep}] {pr(mo)} ({time.time() - t0:.0f}s)", flush=True)
        multilora.save(ck / f"adapters_epoch{ep}.pt", lm.model, {"key": semwm.Key.shipped().label(), "epoch": ep,
                                                                  "adapter_index": "prev_bin (mask id)", "method": "token-level push distillation",
                                                                  "monitor": {x: y for x, y in mo.items() if x != "samples"}})
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "distill_log.json").write_text(json.dumps(log, indent=1, default=float))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--corpus", default="runs/ss_corpus")
    t.add_argument("--out", default="runs/ss_distill_token/teacher")
    t.add_argument("--shard", type=int, default=0)
    t.add_argument("--num-shards", type=int, default=1)
    t.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("train")
    s.add_argument("--teacher", default="runs/ss_distill_token/teacher")
    s.add_argument("--val", default="runs/ss_corpus_val")
    s.add_argument("--out", default="runs/ss_distill_token")
    s.add_argument("--ckpt-dir", default="/l/users/daniil.orel/wmworld_rl/distill_token")
    s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--accum", type=int, default=8)
    s.add_argument("--group", type=int, default=16)
    s.add_argument("--eval-n", type=int, default=48)
    a = ap.parse_args()
    teacher(a) if a.cmd == "teacher" else train(a)


if __name__ == "__main__":
    main()
