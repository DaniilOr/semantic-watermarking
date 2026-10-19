"""RL for the semantic watermark (amendment 13): train LoRA adapters on the 3B so its sentences land in the mask AND stay
meaningful.  Group-relative policy gradient (GRPO-style, one update per batch) with a KL penalty to the frozen base.

Episodes are single sentences in real chain contexts (runs/ss_corpus: prefix + the unit's mask).  Policy = frozen
Llama-3.2-3B + 8 LoRA adapters (wmworld/multilora.py), adapter index = the mask's id (the previous sentence's bin; 8
possible masks).  Scope as elsewhere: prefix[:-1] by the base, adapter from the last prefix token onward.
Per context, G sentences are sampled from the policy with the 3B's own sampler (temperature 0.8, top-p 0.95).
Reward per sentence:
  depth   clip(depth inside the mask / 0.05, -1, 1): > 0 iff the bin is in the mask; grows with distance from the planes
  judge   local judge (Qwen3.5-0.8B, non-thinking; a different family from the policy and from the evaluation judge
          gemma-4-31b, so the reward is not the evaluation metric): expected 1-5 rating of the sentence as a natural, coherent, relevant
          continuation of the context, from its probabilities for the digits 1-5; mapped to (score - 3) / 2
  guards  -1 if the sentence has no proper ending (hit the 64-token cap) or repeats a 4-gram of itself or of the context
  R = w_depth * depth + w_judge * judge + guards
Advantage = group-normalised R.  Loss = -mean_i A_i * mean_t log pi(x_it) + beta * mean_tokens KL_k3(pi || base).
log pi uses the tempered logits (temperature 0.8) without the top-p truncation.
Monitoring every --eval-every steps on fixed validation contexts (runs/ss_corpus_val): in-mask rate, clear-of-0.02 rate,
mean depth, judge score, KL per sentence; the base (adapters off) measured once at step 0.  Sample sentences saved.
"""

from __future__ import annotations

import argparse
import glob
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("steered_semstamp", ROOT / "scripts" / "steered_semstamp.py")
ss = importlib.util.module_from_spec(spec); sys.modules["steered_semstamp"] = ss; spec.loader.exec_module(ss)

from wmworld import multilora, semwm  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM  # noqa: E402
from wmworld.records import write_json  # noqa: E402
from wmworld.textnorm import seed_from  # noqa: E402

QWEN = "/home/daniil.orel/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
JUDGE_INSTR = ("Below is the beginning of a text and a candidate next sentence.\n\nText so far:\n{ctx}\n\nNext sentence:\n{sent}\n\n"
               "Rate the next sentence from 1 to 5 for how natural, coherent and relevant it is as the continuation of the text "
               "(5 = a perfectly natural continuation; 1 = incoherent, off-topic, repetitive or broken). Answer with a single digit.")


class ApiJudge:
    """Judge through OpenRouter (default google/gemma-3-27b-it: a different Gemma generation from the evaluation judge
    gemma-4-31b).  One request per sentence, run in parallel; the score is the expected 1-5 rating from the top-5
    log-probabilities of the first answer token (digits only, renormalised), else the answered digit.  Retries with
    backoff; a sentence whose request fails 6 times gets the neutral score 3 and is counted.  The key is read from
    $OPENROUTER_API_KEY or ~/.config/wmworld/openrouter.key and never printed."""
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, workers: int = 32):
        import os
        self.model, self.workers = model, workers
        k = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not k:
            f = Path.home() / ".config" / "wmworld" / "openrouter.key"
            k = f.read_text().strip() if f.exists() else ""
        if not k:
            raise SystemExit("no OpenRouter key")
        self.key, self.failures, self.calls = k, 0, 0

    def _one(self, ctx: str, sent: str) -> float:
        import requests
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 5,
                "messages": [{"role": "user", "content": JUDGE_INSTR.format(ctx=ctx, sent=sent)}]}
        for attempt in range(6):
            try:
                r = requests.post(self.URL, json=body, timeout=60, headers={"Authorization": f"Bearer {self.key}", "X-Title": "wmworld rl judge"})
                if r.status_code == 200:
                    ch = r.json()["choices"][0]
                    top = ((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
                    dig = {t["token"].strip(): t["logprob"] for t in top if t.get("token", "").strip() in ("1", "2", "3", "4", "5")}
                    if dig:
                        w = {k: math.exp(v) for k, v in dig.items()}
                        return sum(int(k) * v for k, v in w.items()) / sum(w.values())
                    txt = (ch["message"].get("content") or "").strip()[:1]
                    if txt in ("1", "2", "3", "4", "5"):
                        return float(txt)
            except Exception:
                pass
            time.sleep(min(30, 2 ** attempt))
        self.failures += 1
        return 3.0

    def score(self, contexts: list[str], sents: list[str]) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor
        self.calls += len(sents)
        with ThreadPoolExecutor(self.workers) as ex:
            return np.array(list(ex.map(self._one, contexts, sents)), dtype=float)


PARA_INSTR = ("Rewrite the sentence below with different wording while preserving its meaning, facts, names and numbers. "
              "Keep it a single sentence of similar length. Output only the rewritten sentence.\n\nSentence: {sent}")
PARA = None                                                                # set in main() when --para-model is given


class ApiParaphraser:
    """Sentence paraphrases through OpenRouter for the paraphrase reward (run 3b; the evaluation attacker is gemma-4-31b,
    so training uses another model, default google/gemma-3-12b-it).  Temperature 0.7 as the evaluation's paraphrases;
    one request per sentence in parallel, retries with backoff; a failed sentence returns None and is counted."""

    def __init__(self, model: str, workers: int = 32):
        self.model, self.workers, self.key = model, workers, ApiJudge(model).key
        self.failures, self.calls = 0, 0

    def _one(self, sent: str):
        import requests
        body = {"model": self.model, "temperature": 0.7, "max_tokens": 160,
                "messages": [{"role": "user", "content": PARA_INSTR.format(sent=sent)}]}
        for attempt in range(6):
            try:
                r = requests.post(ApiJudge.URL, json=body, timeout=60, headers={"Authorization": f"Bearer {self.key}", "X-Title": "wmworld rl paraphrase"})
                if r.status_code == 200:
                    txt = (r.json()["choices"][0]["message"].get("content") or "").strip().strip('"').strip()
                    if txt:
                        return txt
            except Exception:
                pass
            time.sleep(min(30, 2 ** attempt))
        self.failures += 1
        return None

    def rewrite(self, sents: list[str]) -> list:
        from concurrent.futures import ThreadPoolExecutor
        self.calls += len(sents)
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, sents))


def load_contexts(run: str, prompts: dict[str, str], tok) -> list[dict]:
    corpus = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); corpus[r["prompt_id"]] = r
    items = []
    for pid, c in sorted(corpus.items()):
        chat = tok.apply_chat_template([{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": prompts[pid]}],
                                       tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024")
        prefix = [int(t) for t in tok(chat, add_special_tokens=False)["input_ids"]]
        gen_start = len(prefix)
        for u in c["units"]:
            if u["mask"] is not None:
                items.append({"pid": pid, "unit": u["unit"], "prefix": list(prefix), "gen_start": gen_start, "mask": tuple(u["mask"]),
                              "prev_bin": int(u["prev_bin"]), "next_ids": u["cands"][u["chosen"]]["ids"]})
            prefix += u["cands"][u["chosen"]]["ids"]
    return items


class Judge:
    """Qwen3.5-0.8B loaded text-only (as wmworld/runtime.py).  RIGHT padding and the logits at each row's last real
    token: its linear-attention (DeltaNet) layers would carry left padding into their state."""

    def __init__(self, device: str):
        from transformers import Qwen3_5ForCausalLM
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        self.model = Qwen3_5ForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.digits = [self.tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
        assert all(isinstance(d, int) and d >= 0 for d in self.digits)
        self.tok.padding_side = "right"

    @torch.no_grad()
    def score(self, contexts: list[str], sents: list[str], bs: int = 16) -> np.ndarray:
        """Expected 1-5 rating per (context, sentence)."""
        texts = [self.tok.apply_chat_template([{"role": "user", "content": JUDGE_INSTR.format(ctx=c, sent=s)}], tokenize=False,
                                              add_generation_prompt=True) for c, s in zip(contexts, sents)]
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True).to(self.device)
            last = enc["attention_mask"].sum(-1) - 1
            logits = self.model(**enc).logits[torch.arange(len(last), device=self.device), last][:, self.digits].float()
            out.append((torch.softmax(logits, -1) * torch.arange(1, 6, device=self.device)).sum(-1).cpu().numpy())
        return np.concatenate(out)


def token_logprobs(model, ctrl, prefix: list[int], cands: list[list[int]], rows: list[int], T: float, pad_id: int, device, grad: bool):
    """Per-token log-probabilities [R, L] (tempered, untruncated) of each candidate under its row's adapter, and the mask."""
    pre, last = prefix[:-1], prefix[-1]
    R = len(cands)
    ctrl.set_rows(None)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([pre], device=device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(R)
    Lc = max(len(c) for c in cands)
    inp = torch.full((R, Lc), pad_id, dtype=torch.long, device=device)
    tgt = torch.zeros((R, Lc), dtype=torch.long, device=device)
    msk = torch.zeros((R, Lc), dtype=torch.bool, device=device)
    att = torch.zeros((R, len(pre) + Lc), dtype=torch.long, device=device)
    att[:, :len(pre)] = 1
    for r, c in enumerate(cands):
        seq = [last] + c[:-1]
        inp[r, :len(seq)] = torch.tensor(seq, device=device); tgt[r, :len(c)] = torch.tensor(c, device=device)
        msk[r, :len(c)] = True; att[r, len(pre):len(pre) + len(seq)] = 1
    ctrl.set_rows(rows)
    try:
        with torch.set_grad_enabled(grad):
            logits = model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).logits
            lp = F.log_softmax(logits.float() / T, dim=-1).gather(-1, tgt[..., None]).squeeze(-1)
    finally:
        ctrl.set_rows(None)
    return lp, msk


def repeats(ids: list[int], context: list[int], n_self: int = 4, n_copy: int = 8) -> bool:
    """A 4-gram repeated inside the sentence itself, or >= 8 consecutive tokens copied verbatim from the context.
    (Sharing a 4-gram with the context is normal text -- 'as well as', 'the United States' -- and is not penalised.)"""
    own = [tuple(ids[i:i + n_self]) for i in range(len(ids) - n_self + 1)]
    ctx8 = {tuple(context[i:i + n_copy]) for i in range(len(context) - n_copy + 1)}
    return len(own) != len(set(own)) or any(tuple(ids[i:i + n_copy]) in ctx8 for i in range(len(ids) - n_copy + 1))


def rollout(lm, ctrl, emb, planes, judge, it, G, sampling, unit_a, gen, a, adapter_on=True):
    """G sentences for one context; returns per-sentence dicts with rewards and components."""
    rows = [it["prev_bin"] if adapter_on else -1] * G
    cands, _ = ss.adapter_candidates(lm, ctrl, it["prefix"], rows, sampling, unit_a, 64, 0, 10 ** 6, gen)
    cands = [c for c in cands if c["ids"]]
    if not cands:
        return []
    P = planes.projections(emb.encode([c["text"] for c in cands]))
    dep = np.array([semwm.signed_distance(q, it["mask"], planes) for q in P])
    ctx_text = lm.tokenizer.decode(it["prefix"][it["gen_start"]:], skip_special_tokens=True)[-1200:] or "(start of the text)"
    js = judge.score([ctx_text] * len(cands), [c["text"] for c in cands])
    pdep = [None] * len(cands)
    if PARA is not None:                                                  # paraphrase reward: depth of the rewritten sentence
        para = PARA.rewrite([c["text"] for c in cands])
        ok = [i for i, t in enumerate(para) if t]
        if ok:
            Pp = planes.projections(emb.encode([para[i] for i in ok]))
            for i, q in zip(ok, Pp):
                pdep[i] = semwm.signed_distance(q, it["mask"], planes)
    out = []
    for c, q, d, j, pd in zip(cands, P, dep, js, pdep):
        noend = (not c["found"] and not c["ends"])
        rep_ = repeats(c["ids"], it["prefix"][it["gen_start"]:])
        guard = -1.0 if (noend or rep_) else 0.0
        r = a.w_depth * float(np.clip(d / 0.05, -1, 1)) + a.w_judge * (float(j) - 3) / 2 + guard
        if pd is not None:
            r += a.w_para * float(np.clip(pd / 0.05, -1, 1))
        out.append({"ids": c["ids"], "text": c["text"], "depth": float(d), "in_mask": bool(d > 0), "clear": bool(d >= semwm.MARGIN),
                    "judge": float(j), "guard": guard, "noend": bool(noend), "repeat": bool(rep_), "reward": r,
                    "para_depth": None if pd is None else float(pd)})
    return out


@torch.no_grad()
def monitor(lm, ctrl, emb, planes, judge, items, G, sampling, unit_a, a, pad_id, adapter_on: bool, step: int) -> dict:
    gen = torch.Generator(device=lm.device)
    rs, kls, samples = [], [], []
    for it in items:
        gen.manual_seed(seed_from("rl_monitor", it["pid"], it["unit"]))
        ro = rollout(lm, ctrl, emb, planes, judge, it, G, sampling, unit_a, gen, a, adapter_on)
        if not ro:
            continue
        rs += ro
        if adapter_on:
            cands = [x["ids"] for x in ro]
            lp_pol, m = token_logprobs(lm.model, ctrl, it["prefix"], cands, [it["prev_bin"]] * len(cands), a.temperature, pad_id, lm.device, False)
            lp_ref, _ = token_logprobs(lm.model, ctrl, it["prefix"], cands, [-1] * len(cands), a.temperature, pad_id, lm.device, False)
            kls += ((lp_pol - lp_ref) * m).sum(-1).tolist()                     # sampled KL(pi || base) per sentence
        if len(samples) < 12:
            samples.append({"context_tail": lm.tokenizer.decode(it["prefix"][-60:], skip_special_tokens=True), "sentence": ro[0]["text"],
                            "depth": ro[0]["depth"], "judge": ro[0]["judge"]})
    f = lambda k: float(np.mean([x[k] for x in rs]))
    return {"step": step, "adapter": adapter_on, "n": len(rs), "in_mask": f("in_mask"), "clear_0.02": f("clear"), "depth": f("depth"),
            "judge": f("judge"), "guard_rate": float(np.mean([x["guard"] < 0 for x in rs])), "noend_rate": f("noend"), "repeat_rate": f("repeat"),
            "reward": f("reward"),
            "kl_per_sentence": float(np.mean(kls)) if kls else 0.0, "samples": samples,
            **({} if PARA is None else {
                "para_in_mask": float(np.mean([x["para_depth"] > 0 for x in rs if x["para_depth"] is not None] or [np.nan])),
                "para_clear_0.02": float(np.mean([x["para_depth"] >= semwm.MARGIN for x in rs if x["para_depth"] is not None] or [np.nan])),
                "para_failed": int(sum(x["para_depth"] is None for x in rs))})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/ss_corpus")
    ap.add_argument("--val", default="runs/ss_corpus_val")
    ap.add_argument("--out", default="runs/ss_rl/v1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--contexts-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.05, help="KL penalty coefficient (per token)")
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-judge", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--judge", choices=["local", "api"], default="local", help="local Qwen3.5-0.8B or an OpenRouter model")
    ap.add_argument("--judge-model", default="google/gemma-3-27b-it")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--ckpt-dir", default="", help="where adapter checkpoints go (default: --out)")
    ap.add_argument("--stop-patience", type=int, default=2,
                    help="stop after this many stale monitors in a row (see --stale-rule); a new best clear-in-mask share resets the count")
    ap.add_argument("--stale-rule", choices=["kl", "any"], default="kl",
                    help="kl (runs v1/v2): a monitor is stale if the clear share did not beat its best AND drift (KL) rose; "
                         "any (run 3): every monitor without a new best clear share is stale")
    ap.add_argument("--para-model", default="", help="paraphrase reward: OpenRouter model that rewrites every sampled sentence (off if empty)")
    ap.add_argument("--w-para", type=float, default=1.0, help="weight of the depth-after-paraphrase term")
    ap.add_argument("--kl-cap", type=float, default=0.0, help="hard drift cap: sampled log(pi/base) per sentence above this (nats) is penalised (0 = off)")
    ap.add_argument("--kl-cap-penalty", type=float, default=2.0, help="reward subtracted from a sentence over the cap")
    ap.add_argument("--init-adapters", default="", help="start from these adapters (e.g. RFT round 3) instead of zero-initialised B")
    a = ap.parse_args()
    t0 = time.time()
    random.seed(0); torch.manual_seed(0)
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    lm = ss.LM(m["path"], "bf16", device="cuda:0")
    if a.init_adapters:
        ctrl, init_meta = multilora.load(a.init_adapters, lm.model)
        assert init_meta["key"] == semwm.Key.shipped().label(), f"init adapters trained for key {init_meta['key']}"
        print(f"initialised from {a.init_adapters}", flush=True)
    else:
        ctrl = multilora.attach(lm.model, n=8, r=a.r, alpha=a.alpha)
    params = multilora.trainable(lm.model)
    for q in params:
        q.requires_grad_(True)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=64)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    judge = ApiJudge(a.judge_model) if a.judge == "api" else Judge("cuda:0")
    global PARA
    if a.para_model:
        PARA = ApiParaphraser(a.para_model)
        t_p = time.time(); probe = PARA.rewrite(["The committee approved the new budget after a long debate on Tuesday."] * 16)
        print(f"[paraphraser] {a.para_model}: 16 parallel calls in {time.time() - t_p:.1f}s, failed {sum(x is None for x in probe)}; "
              f"example: {next((x for x in probe if x), None)!r}", flush=True)
        if sum(x is None for x in probe) > 1:
            raise SystemExit("paraphraser unreliable from this node")
    sampling = SimpleNamespace(temperature=a.temperature, top_p=0.95)
    unit_a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, W=16)
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    train, val = load_contexts(a.train, prompts, lm.tokenizer), load_contexts(a.val, prompts, lm.tokenizer)
    val_mon = random.Random(1).sample(val, min(a.eval_n, len(val)))
    print(f"train contexts {len(train)}, monitor contexts {len(val_mon)}; {sum(p.numel() for p in params) / 1e6:.1f}M trainable ({time.time() - t0:.0f}s)", flush=True)
    # reliability of the reward's judge: real next sentence vs a sentence from another text, same contexts
    from wmworld.posstats import auroc
    ctxs = [lm.tokenizer.decode(it["prefix"][it["gen_start"]:], skip_special_tokens=True)[-1200:] or "(start of the text)" for it in val_mon]
    real = [lm.tokenizer.decode(it["next_ids"], skip_special_tokens=True).strip() for it in val_mon]
    other = real[1:] + real[:1]
    s_real, s_other = judge.score(ctxs, real), judge.score(ctxs, other)
    judge_check = {"mean_real": float(s_real.mean()), "mean_mismatched": float(s_other.mean()), "auroc_real_vs_mismatched": auroc(list(s_real), list(s_other))}
    print(f"[judge check] real next sentence {judge_check['mean_real']:.3f} vs sentence from another text {judge_check['mean_mismatched']:.3f}; "
          f"AUROC {judge_check['auroc_real_vs_mismatched']:.3f}", flush=True)
    if a.judge == "api":
        judge_check.update({"api_calls": judge.calls, "api_failures": judge.failures})
        print(f"[judge check] api calls {judge.calls}, failed {judge.failures}", flush=True)
        if judge.failures > 0.1 * judge.calls:
            raise SystemExit("API judge unreliable from this node (more than 10% of calls failed)")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    meta = {"key": semwm.Key.shipped().label(), "adapter_index": "prev_bin (mask id)", "args": vars(a),
            "judge": a.judge_model if a.judge == "api" else QWEN}
    log = {"meta": meta, "monitor": [], "train": [], "judge_check": judge_check}
    base = monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, a, lm.pad_id, False, 0)
    log["monitor"].append(base)
    print(f"[monitor base] in-mask {base['in_mask']:.3f} clear {base['clear_0.02']:.3f} depth {base['depth']:+.4f} judge {base['judge']:.3f} "
          f"guard {base['guard_rate']:.3f} (no ending {base['noend_rate']:.3f}, repeat {base['repeat_rate']:.3f}) reward {base['reward']:+.3f}"
          + (f" | after paraphrase: in-mask {base['para_in_mask']:.3f} clear {base['para_clear_0.02']:.3f} (failed {base['para_failed']})" if PARA else ""), flush=True)
    ckpt_dir = Path(a.ckpt_dir or a.out); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_clear, stale = base["clear_0.02"], 0
    if a.init_adapters:                                  # the starting adapters are the bar to beat, not the base model
        init = monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, a, lm.pad_id, True, 0)
        init["init_adapters"] = a.init_adapters
        log["monitor"].append(init)
        best_clear = init["clear_0.02"]
        print(f"[monitor init] in-mask {init['in_mask']:.3f} clear {init['clear_0.02']:.3f} judge {init['judge']:.3f} "
              f"kl/sent {init['kl_per_sentence']:.3f}", flush=True)
    gen = torch.Generator(device=lm.device)
    order = random.Random(2).sample(range(len(train)), len(train))
    acc = {k: [] for k in ("reward", "in_mask", "depth", "judge", "kl", "over_cap")}
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        n_rows = 0
        for k in range(a.contexts_per_step):
            it = train[order[((step - 1) * a.contexts_per_step + k) % len(order)]]
            gen.manual_seed(seed_from("rl", step, k))
            with torch.no_grad():
                ro = rollout(lm, ctrl, emb, planes, judge, it, a.group, sampling, unit_a, gen, a, True)
            if len(ro) < 2:
                continue
            R = np.array([x["reward"] for x in ro])
            cands = [x["ids"] for x in ro]
            with torch.no_grad():
                lp_ref, _ = token_logprobs(lm.model, ctrl, it["prefix"], cands, [-1] * len(cands), a.temperature, lm.pad_id, lm.device, False)
            lp, msk = token_logprobs(lm.model, ctrl, it["prefix"], cands, [it["prev_bin"]] * len(cands), a.temperature, lm.pad_id, lm.device, True)
            mf = msk.float(); ntok = mf.sum(-1).clamp_min(1)
            if a.kl_cap > 0:                                               # hard drift cap: a sentence whose sampled
                kl_i = ((lp.detach() - lp_ref) * mf).sum(-1).cpu().numpy()  # log(pi/base) exceeds the cap is penalised
                over = kl_i > a.kl_cap
                R = R - a.kl_cap_penalty * over
                acc["over_cap"].append(float(over.mean()))
            A = (R - R.mean()) / (R.std() + 1e-4)
            seq_lp = (lp * mf).sum(-1) / ntok
            d = (lp_ref - lp) * mf
            k3 = ((torch.exp(d) - d - 1) * mf).sum() / mf.sum()                     # KL(pi || base), k3 estimator
            loss = -(torch.tensor(A, device=lm.device, dtype=torch.float32) * seq_lp).mean() + a.beta * k3
            (loss / a.contexts_per_step).backward()
            n_rows += len(ro)
            acc["reward"].append(R.mean()); acc["in_mask"].append(np.mean([x["in_mask"] for x in ro])); acc["depth"].append(np.mean([x["depth"] for x in ro]))
            acc["judge"].append(np.mean([x["judge"] for x in ro])); acc["kl"].append(float(((lp.detach() - lp_ref) * mf).sum(-1).mean()))
        if n_rows:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if step % 20 == 0:
            s = {k: float(np.mean(v)) for k, v in acc.items() if v}
            log["train"].append(s | {"step": step})
            print(f"[train] step {step} reward {s.get('reward', 0):+.3f} in-mask {s.get('in_mask', 0):.3f} depth {s.get('depth', 0):+.4f} "
                  f"judge {s.get('judge', 0):.3f} kl/sent {s.get('kl', 0):.3f}" + (f" over-cap {s.get('over_cap', 0):.3f}" if a.kl_cap > 0 else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            acc = {k: [] for k in acc}
        if step % a.eval_every == 0 or step == a.steps:
            mo = monitor(lm, ctrl, emb, planes, judge, val_mon, a.group, sampling, unit_a, a, lm.pad_id, True, step)
            prev_kl = log["monitor"][-1].get("kl_per_sentence", 0.0)
            log["monitor"].append(mo)
            if mo["clear_0.02"] > best_clear:
                best_clear, stale = mo["clear_0.02"], 0
            elif a.stale_rule == "any" or mo["kl_per_sentence"] > prev_kl:
                stale += 1
            print(f"[monitor {step}] in-mask {mo['in_mask']:.3f} clear {mo['clear_0.02']:.3f} depth {mo['depth']:+.4f} judge {mo['judge']:.3f} "
                  f"guard {mo['guard_rate']:.3f} (no ending {mo['noend_rate']:.3f}, repeat {mo['repeat_rate']:.3f}) kl/sent {mo['kl_per_sentence']:.3f} "
                  f"reward {mo['reward']:+.3f}" + (f" | api judge calls {judge.calls}, failed {judge.failures}" if a.judge == "api" else "")
                  + (f" | after paraphrase: in-mask {mo['para_in_mask']:.3f} clear {mo['para_clear_0.02']:.3f} (failed {mo['para_failed']})" if PARA else ""), flush=True)
            write_json(Path(a.out) / "rl_log.json", log)
        if step % a.save_every == 0 or step == a.steps:
            last_mo = log["monitor"][-1] if log["monitor"][-1]["step"] == step else None
            multilora.save(ckpt_dir / f"adapters_step{step}.pt", lm.model, meta | {"step": step, "monitor": {k: v for k, v in (last_mo or {}).items() if k != "samples"}})
            print(f"  saved {ckpt_dir / f'adapters_step{step}.pt'}", flush=True)
        if step % a.eval_every == 0 and stale >= a.stop_patience:
            print(f"[stop] {stale} stale monitors (rule {a.stale_rule}) without beating the best clear-in-mask share {best_clear:.3f}", flush=True)
            log["stopped_early"] = {"step": step, "best_clear": best_clear}
            write_json(Path(a.out) / "rl_log.json", log)
            break
    print(f"done ({time.time() - t0:.0f}s); peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GiB reserved", flush=True)


if __name__ == "__main__":
    main()
