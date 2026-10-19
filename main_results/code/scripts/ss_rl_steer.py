"""RL-trained steering head (amendment 14): a small network that biases the frozen 3B's next-token choices so the
sentence lands in the mask and stays meaningful.  Same GRPO setup, reward, monitors and stop rule as
scripts/ss_rl_mask.py (the LoRA run), but the 3B is untouched and only the head is trained.

Per token: base logits l (3B, frozen) -> s = l / T; nucleus N = top-p(0.95) set of softmax(s); the head scores the 64
most likely LEXICAL tokens of N (EOS / punctuation / whitespace -- semwm.boundary_token_mask -- are never biased):
  b_v = cap * tanh(head(h_t, e_v, mask_id)),  cap = 2 logits,  head's last layer zero-initialised (step 0 == base)
  s'_v = s_v + b_v - c on the lexical tokens of N, c chosen so their total mass is unchanged (P(boundary) exact)
  pi = softmax over N of s'  (tokens outside the nucleus keep probability 0, as in the base sampler)
KL(pi || base nucleus distribution) is computed exactly per position.  Sampling and the gradient pass compute the same
distribution; gradients flow only into the head (the 3B runs forward only, with a teacher-forced pass for the update).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ss_rl_mask", ROOT / "scripts" / "ss_rl_mask.py")
rl = importlib.util.module_from_spec(spec); sys.modules["ss_rl_mask"] = rl; spec.loader.exec_module(rl)
ss = rl.ss

from wmworld import semwm  # noqa: E402
from wmworld.posstats import auroc  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import seed_from  # noqa: E402

TOPK = 64


class SteerHead(nn.Module):
    def __init__(self, d: int, n_masks: int = 8, hidden: int = 512, cap: float = 2.0):
        super().__init__()
        self.norm_h, self.norm_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mask_emb = nn.Embedding(n_masks, 64)
        self.net = nn.Sequential(nn.Linear(2 * d + 64, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.cap = cap

    def forward(self, h: torch.Tensor, e: torch.Tensor, mask_id: torch.Tensor) -> torch.Tensor:
        """h [N, d], e [N, K, d], mask_id [N] -> bias [N, K]."""
        N, K, _ = e.shape
        hh = self.norm_h(h.float())[:, None, :].expand(N, K, -1)
        mm = self.mask_emb(mask_id)[:, None, :].expand(N, K, -1)
        x = torch.cat([hh, self.norm_e(e.float()), mm], dim=-1)
        return self.cap * torch.tanh(self.net(x).squeeze(-1))


def steered_logprobs(logits: torch.Tensor, h: torch.Tensor, mask_id: torch.Tensor, head: SteerHead, emb_w, boundary, T: float, top_p: float,
                     eos_block: torch.Tensor | None = None):
    """logits [N, V] (raw), h [N, d] -> (log pi [N, V] with -inf outside the nucleus, log p_nucleus [N, V])."""
    s = logits.float() / T
    if eos_block is not None:
        s = s.masked_fill(eos_block, float("-inf"))
    lp_base = ss.processed_logprobs(s * T, T, top_p, None)                       # base sampler (nucleus renormalised)
    in_nuc = torch.isfinite(lp_base)
    lex = in_nuc & ~boundary[None, :]
    score = torch.where(lex, lp_base, torch.full_like(lp_base, float("-inf")))
    K = min(TOPK, s.shape[-1])
    top_lp, top_idx = torch.topk(score, K, dim=-1)                               # most likely lexical nucleus tokens
    valid = torch.isfinite(top_lp)
    bias = head(h, emb_w[top_idx], mask_id) * valid
    s_n = torch.where(in_nuc, s, torch.full_like(s, float("-inf")))
    s_b = s_n.scatter_add(-1, top_idx, bias)
    # mass-preserving tilt of the lexical class.  Masked with a FINITE fill so no -inf - -inf appears anywhere in the
    # graph (torch.where does not stop NaN gradients from the unselected branch); rows without lexical tokens get c = 0.
    fill = torch.full_like(s, -1e4)
    lse = lambda x: torch.logsumexp(torch.where(lex, torch.where(torch.isfinite(x), x, fill), fill), dim=-1, keepdim=True)
    c = lse(s_b) - lse(s_n)
    s_b = torch.where(lex, s_b - c, s_b)
    return torch.log_softmax(s_b, dim=-1), lp_base


@torch.no_grad()
def steered_candidates(lm, head, emb_w, boundary, prefix, mask_id, n, sampling, unit_a, gen):
    """n sentences sampled with the steering head (same stop rule as adapter_candidates)."""
    eos = set(lm.eos_ids)
    cache = lm.model.model(input_ids=torch.tensor([prefix[:-1]], device=lm.device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(n)
    cur = torch.full((n, 1), prefix[-1], dtype=torch.long, device=lm.device)
    out, alive, saw = [[] for _ in range(n)], [True] * n, [False] * n
    mid = torch.full((n,), mask_id, dtype=torch.long, device=lm.device)
    eos_t = torch.tensor(sorted(eos), device=lm.device)
    for t in range(unit_a.sent_max_tokens):
        hs = lm.model.model(input_ids=cur, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
        logits = lm.model.lm_head(hs)
        block = torch.zeros_like(logits, dtype=torch.bool)
        if t < 2:
            block[:, eos_t] = True
        lp, _ = steered_logprobs(logits, hs, mid, head, emb_w, boundary, sampling.temperature, sampling.top_p, block)
        nxt = torch.multinomial(lp.exp(), 1, generator=gen)
        for r in range(n):
            if not alive[r]:
                continue
            tk = int(nxt[r, 0])
            if tk in eos:
                alive[r], saw[r] = False, True; continue
            out[r].append(tk)
            if len(out[r]) >= 4 and semwm.first_boundary(lm.tokenizer.decode(out[r], skip_special_tokens=True)) is not None:
                alive[r] = False
        if not any(alive):
            break
        cur = nxt
    cands = []
    for r in range(n):
        cid, text, found = semwm.trim_candidate(lm.tokenizer, out[r], [])
        if cid and text:
            cands.append({"ids": cid, "text": text, "ends": saw[r] and not found, "found": bool(found)})
    return cands


def policy_terms(lm, head, emb_w, boundary, prefix, cands, mask_id, sampling):
    """Teacher-forced: mean log pi per candidate (with grad into the head) and the exact mean per-token KL(pi || base)."""
    R = len(cands)
    with torch.no_grad():
        cache = lm.model.model(input_ids=torch.tensor([prefix[:-1]], device=lm.device), use_cache=True).past_key_values
        cache.batch_repeat_interleave(R)
        Lc = max(len(c) for c in cands)
        inp = torch.full((R, Lc), lm.pad_id, dtype=torch.long, device=lm.device)
        att = torch.zeros((R, len(prefix) - 1 + Lc), dtype=torch.long, device=lm.device); att[:, :len(prefix) - 1] = 1
        for r, c in enumerate(cands):
            seq = [prefix[-1]] + c[:-1]
            inp[r, :len(seq)] = torch.tensor(seq, device=lm.device); att[r, len(prefix) - 1:len(prefix) - 1 + len(seq)] = 1
        H = lm.model.model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state
        logits = lm.model.lm_head(H)
    rows, poss, toks = [], [], []
    for r, c in enumerate(cands):
        for t, x in enumerate(c):
            rows.append(r); poss.append(t); toks.append(x)
    rows_t, pos_t = torch.tensor(rows, device=lm.device), torch.tensor(poss, device=lm.device)
    block = torch.zeros((len(rows), logits.shape[-1]), dtype=torch.bool, device=lm.device)
    block[pos_t < 2] = False
    eos_t = torch.tensor(lm.eos_ids, device=lm.device)
    early = (pos_t < 2).nonzero().flatten()
    if len(early):
        block[early[:, None], eos_t[None, :]] = True
    lp, lp_base = steered_logprobs(logits[rows_t, pos_t], H[rows_t, pos_t], torch.full((len(rows),), mask_id, dtype=torch.long, device=lm.device),
                                   head, emb_w, boundary, sampling.temperature, sampling.top_p, block)
    tok_lp = lp.gather(-1, torch.tensor(toks, device=lm.device)[:, None]).squeeze(-1)
    fin = torch.isfinite(lp_base)                                                     # the nucleus (pi's support)
    zero = torch.zeros_like(lp)
    lp_s, lb_s = torch.where(fin, lp, zero), torch.where(fin, lp_base, zero)          # finite inputs: no NaN gradients
    kl = (lp_s.exp() * fin * (lp_s - lb_s)).sum(-1)
    seq_lp = torch.zeros(R, device=lm.device).index_add(0, rows_t, tok_lp) / torch.tensor([len(c) for c in cands], device=lm.device, dtype=torch.float32)
    return seq_lp, kl.mean(), kl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/ss_corpus")
    ap.add_argument("--val", default="runs/ss_corpus_val")
    ap.add_argument("--out", default="runs/ss_rl/steer_v1")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--contexts-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-judge", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--stop-patience", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    random.seed(0); torch.manual_seed(0)
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    lm = ss.LM(m["path"], "bf16", device="cuda:0")
    for p_ in lm.model.parameters():
        p_.requires_grad_(False)
    emb_w = lm.model.get_input_embeddings().weight
    boundary = semwm.boundary_token_mask(lm.tokenizer, lm.eos_ids + [lm.pad_id], lm.device)
    head = SteerHead(lm.model.config.hidden_size, cap=a.cap).to(lm.device)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=64)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    judge = rl.Judge("cuda:0")
    sampling = SimpleNamespace(temperature=a.temperature, top_p=0.95)
    unit_a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, W=16)
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    train, val = rl.load_contexts(a.train, prompts, lm.tokenizer), rl.load_contexts(a.val, prompts, lm.tokenizer)
    val_mon = random.Random(1).sample(val, min(a.eval_n, len(val)))                 # same monitor contexts as the LoRA runs
    print(f"train contexts {len(train)}, monitor {len(val_mon)}; head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M params ({time.time() - t0:.0f}s)", flush=True)

    def rollout(it, n, gen):
        cands = steered_candidates(lm, head, emb_w, boundary, it["prefix"], it["prev_bin"], n, sampling, unit_a, gen)
        cands = [c for c in cands if c["ids"]]
        if not cands:
            return []
        P = planes.projections(emb.encode([c["text"] for c in cands]))
        dep = np.array([semwm.signed_distance(q, it["mask"], planes) for q in P])
        ctx_text = lm.tokenizer.decode(it["prefix"][it["gen_start"]:], skip_special_tokens=True)[-1200:] or "(start of the text)"
        js = judge.score([ctx_text] * len(cands), [c["text"] for c in cands])
        out = []
        for c, d, j in zip(cands, dep, js):
            noend = not c["found"] and not c["ends"]; rep_ = rl.repeats(c["ids"], it["prefix"][it["gen_start"]:])
            guard = -1.0 if (noend or rep_) else 0.0
            out.append({"ids": c["ids"], "text": c["text"], "depth": float(d), "in_mask": bool(d > 0), "clear": bool(d >= semwm.MARGIN),
                        "judge": float(j), "guard": guard, "noend": bool(noend), "repeat": bool(rep_),
                        "reward": a.w_depth * float(np.clip(d / 0.05, -1, 1)) + a.w_judge * (float(j) - 3) / 2 + guard})
        return out

    @torch.no_grad()
    def monitor(step):
        gen = torch.Generator(device=lm.device)
        rs, kls, samples = [], [], []
        for it in val_mon:
            gen.manual_seed(seed_from("rl_monitor", it["pid"], it["unit"]))
            ro = rollout(it, a.group, gen)
            if not ro:
                continue
            rs += ro
            _, _, kl_tok = policy_terms(lm, head, emb_w, boundary, it["prefix"], [x["ids"] for x in ro], it["prev_bin"], sampling)
            kls.append(float(kl_tok.sum()) / len(ro))                                  # KL per sentence
            if len(samples) < 12:
                samples.append({"context_tail": lm.tokenizer.decode(it["prefix"][-60:], skip_special_tokens=True), "sentence": ro[0]["text"],
                                "depth": ro[0]["depth"], "judge": ro[0]["judge"]})
        f = lambda k: float(np.mean([x[k] for x in rs]))
        return {"step": step, "n": len(rs), "in_mask": f("in_mask"), "clear_0.02": f("clear"), "depth": f("depth"), "judge": f("judge"),
                "guard_rate": float(np.mean([x["guard"] < 0 for x in rs])), "noend_rate": f("noend"), "repeat_rate": f("repeat"),
                "reward": f("reward"), "kl_per_sentence": float(np.mean(kls)), "samples": samples}

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
    log = {"meta": {"key": semwm.Key.shipped().label(), "args": vars(a), "judge": rl.QWEN}, "monitor": [], "train": []}
    base = monitor(0); log["monitor"].append(base)
    pr = lambda mo: (f"in-mask {mo['in_mask']:.3f} clear {mo['clear_0.02']:.3f} depth {mo['depth']:+.4f} judge {mo['judge']:.3f} "
                     f"guard {mo['guard_rate']:.3f} (no ending {mo['noend_rate']:.3f}, repeat {mo['repeat_rate']:.3f}) kl/sent {mo['kl_per_sentence']:.3f} reward {mo['reward']:+.3f}")
    print(f"[monitor base] {pr(base)}", flush=True)
    ckpt_dir = Path(a.ckpt_dir or a.out); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_clear, stale = base["clear_0.02"], 0
    gen = torch.Generator(device=lm.device)
    order = random.Random(2).sample(range(len(train)), len(train))
    acc = {k: [] for k in ("reward", "in_mask", "depth", "judge", "kl")}
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for k in range(a.contexts_per_step):
            it = train[order[((step - 1) * a.contexts_per_step + k) % len(order)]]
            gen.manual_seed(seed_from("rl_steer", step, k))
            ro = rollout(it, a.group, gen)
            if len(ro) < 2:
                continue
            R = np.array([x["reward"] for x in ro]); A = (R - R.mean()) / (R.std() + 1e-4)
            seq_lp, kl_mean, kl_tok = policy_terms(lm, head, emb_w, boundary, it["prefix"], [x["ids"] for x in ro], it["prev_bin"], sampling)
            loss = -(torch.tensor(A, device=lm.device, dtype=torch.float32) * seq_lp).mean() + a.beta * kl_mean
            (loss / a.contexts_per_step).backward()
            acc["reward"].append(R.mean()); acc["in_mask"].append(np.mean([x["in_mask"] for x in ro])); acc["depth"].append(np.mean([x["depth"] for x in ro]))
            acc["judge"].append(np.mean([x["judge"] for x in ro])); acc["kl"].append(float(kl_tok.detach().sum()) / len(ro))
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            s_ = {k: float(np.mean(v)) for k, v in acc.items() if v}; log["train"].append(s_ | {"step": step})
            print(f"[train] step {step} reward {s_.get('reward', 0):+.3f} in-mask {s_.get('in_mask', 0):.3f} depth {s_.get('depth', 0):+.4f} "
                  f"judge {s_.get('judge', 0):.3f} kl/sent {s_.get('kl', 0):.3f} ({time.time() - t0:.0f}s)", flush=True)
            acc = {k: [] for k in acc}
        if step % a.eval_every == 0 or step == a.steps:
            prev_kl = log["monitor"][-1]["kl_per_sentence"]
            mo = monitor(step); log["monitor"].append(mo)
            if mo["clear_0.02"] > best_clear:
                best_clear, stale = mo["clear_0.02"], 0
            elif mo["kl_per_sentence"] > prev_kl:
                stale += 1
            print(f"[monitor {step}] {pr(mo)}", flush=True)
            write_json(Path(a.out) / "rl_log.json", log)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"state": head.state_dict(), "cap": a.cap, "step": step, "args": vars(a)}, ckpt_dir / f"steerhead_step{step}.pt")
            print(f"  saved {ckpt_dir / f'steerhead_step{step}.pt'}", flush=True)
        if step % a.eval_every == 0 and stale >= a.stop_patience:
            print(f"[stop] drift rose over {stale} monitors without beating the best clear-in-mask share {best_clear:.3f}", flush=True)
            log["stopped_early"] = {"step": step, "best_clear": best_clear}; write_json(Path(a.out) / "rl_log.json", log)
            break
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
