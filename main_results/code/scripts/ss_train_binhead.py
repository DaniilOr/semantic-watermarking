"""Train the bin head (amendment 12) on the candidate corpus; the 3B stays frozen and runs forward only.

For every candidate c of every context (16 exact 3B samples per sentence position) and every position t of c, the
pair (h_t = final hidden state that predicts c[t], e = embedding of c[t]) is labelled with the REGION of the finished
candidate (wmworld/binhead.py: bin b and clear of the 0.02 margin, or "near a plane").  Cross-entropy, no class
weights (the head must stay calibrated: steering uses its probabilities in Bayes' rule).
Validation (every --eval-every contexts, fixed validation subset): cross-entropy and accuracy overall and by relative
position in the sentence, and AUROC of log P(region in the unit's mask | prefix, next token) for predicting that the
candidate ends in the mask's clear region -- at the first token and at the middle of the sentence.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import binhead, semwm  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM  # noqa: E402
from wmworld.posstats import auroc  # noqa: E402
from wmworld.records import content_hash, write_json  # noqa: E402

DATE = "26 Jul 2024"


def load_items(runs: list[str], prompts: dict[str, str], tok, margin: float) -> list[dict]:
    corpus = {}
    for run in runs:
        for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
            for ln in open(f):
                r = json.loads(ln); corpus[r["prompt_id"]] = r
    items = []
    for pid, c in sorted(corpus.items()):
        chat = tok.apply_chat_template([{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": prompts[pid]}],
                                       tokenize=False, add_generation_prompt=True, date_string=DATE)
        prefix = [int(t) for t in tok(chat, add_special_tokens=False)["input_ids"]]
        for u in c["units"]:
            cd = [x for x in u["cands"] if x["ids"]]
            if u["mask"] is not None and len(cd) >= 2:
                items.append({"pid": pid, "unit": u["unit"], "prefix": list(prefix), "mask": tuple(u["mask"]),
                              "cands": [x["ids"] for x in cd], "labels": [binhead.region_label(x["proj"], margin) for x in cd]})
            prefix += u["cands"][u["chosen"]]["ids"]
    return items


@torch.no_grad()
def hidden_pairs(model, item: dict, pad_id: int, device):
    """Final hidden states H [n_pairs, d] that predict each candidate token, the token ids [n_pairs], the row of each
    pair, and its relative position in the candidate (0 = first token)."""
    pre, last = item["prefix"][:-1], item["prefix"][-1]
    R = len(item["cands"])
    cache = model.model(input_ids=torch.tensor([pre], device=device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(R)
    Lc = max(len(c) for c in item["cands"])
    inp = torch.full((R, Lc), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((R, len(pre) + Lc), dtype=torch.long, device=device)
    att[:, :len(pre)] = 1
    for r, c in enumerate(item["cands"]):
        seq = [last] + c[:-1]
        inp[r, :len(seq)] = torch.tensor(seq, device=device); att[r, len(pre):len(pre) + len(seq)] = 1
    H = model.model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state
    hs, toks, rows, rel = [], [], [], []
    for r, c in enumerate(item["cands"]):
        hs.append(H[r, :len(c)]); toks += c; rows += [r] * len(c); rel += [t / max(1, len(c) - 1) for t in range(len(c))]
    return torch.cat(hs), torch.tensor(toks, device=device), rows, np.array(rel)


def evaluate(model, head, emb_w, items, pad_id, device) -> dict:
    head.eval()
    ce, acc, n = 0.0, 0.0, 0
    by_pos = {k: [0.0, 0] for k in ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")}
    first_s, first_y, mid_s, mid_y = [], [], [], []
    with torch.no_grad():
        for it in items:
            H, toks, rows, rel = hidden_pairs(model, it, pad_id, device)
            logits = head(H, emb_w[toks])
            y = torch.tensor([it["labels"][r] for r in rows], device=device)
            l = F.cross_entropy(logits, y, reduction="none")
            ce += float(l.sum()); acc += float((logits.argmax(-1) == y).sum()); n += len(y)
            for k, (lo, hi) in zip(by_pos, [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]):
                sel = (rel >= lo) & (rel < hi)
                by_pos[k][0] += float(l[torch.tensor(sel, device=device)].sum()); by_pos[k][1] += int(sel.sum())
            ml = binhead.mask_logprob(logits, it["mask"]).cpu().numpy()
            inreg = np.array([it["labels"][r] in it["mask"] for r in rows])
            rows_a = np.array(rows)
            for r in range(len(it["cands"])):
                idx = np.flatnonzero(rows_a == r)
                first_s.append(ml[idx[0]]); first_y.append(inreg[idx[0]])
                mid_s.append(ml[idx[len(idx) // 2]]); mid_y.append(inreg[idx[len(idx) // 2]])
    head.train()
    au = lambda s, y: auroc([x for x, t in zip(s, y) if t], [x for x, t in zip(s, y) if not t])
    return {"ce": ce / n, "acc": acc / n, "pairs": n, "ce_by_position": {k: v[0] / max(1, v[1]) for k, v in by_pos.items()},
            "auroc_mask_region_first_token": au(first_s, first_y), "auroc_mask_region_mid_sentence": au(mid_s, mid_y),
            "share_in_mask_region": float(np.mean(first_y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", default=[])
    ap.add_argument("--val", action="append", default=[])
    ap.add_argument("--out", default="runs/ss_binhead/v1")
    ap.add_argument("--margin", type=float, default=semwm.MARGIN)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0); random.seed(0)
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    tok = AutoTokenizer.from_pretrained(m["path"])
    pad_id = int(tok.convert_tokens_to_ids("<|finetune_right_pad_id|>"))
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    train, val = load_items(a.train, prompts, tok, a.margin), load_items(a.val, prompts, tok, a.margin)
    val_eval = random.Random(1).sample(val, min(a.eval_n, len(val)))
    lab = np.bincount([l for it in train for l in it["labels"]], minlength=binhead.N_REGIONS)
    print(f"train contexts {len(train)}, val contexts {len(val)} (eval {len(val_eval)}); region labels " +
          " ".join(f"{i}:{c}" for i, c in enumerate(lab)) + f" (8 = near a plane)", flush=True)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(m["path"], dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb_w = model.get_input_embeddings().weight
    head = binhead.BinHead(model.config.hidden_size, a.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    total = int(math.ceil(len(train) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    meta = {"key": semwm.Key.shipped().label(), "margin": a.margin, "train_runs": a.train, "val_runs": a.val, "args": vars(a),
            "corpus_hash": content_hash({"train": sorted({it["pid"] for it in train}), "n": len(train)}, length=12), "label_counts": lab.tolist()}
    prior = lab / lab.sum()
    log = {"meta": meta, "evals": [], "ce_of_label_prior": float(-(prior * np.log(prior)).sum())}
    print(f"cross-entropy of the label prior alone: {log['ce_of_label_prior']:.4f}", flush=True)
    order = [i for _ in range(int(math.ceil(a.epochs))) for i in random.Random(2).sample(range(len(train)), len(train))][: int(len(train) * a.epochs)]
    best, run_ce, run_n = float("inf"), 0.0, 0
    for n_ctx, i in enumerate(order, 1):
        it = train[i]
        H, toks, rows, _ = hidden_pairs(model, it, pad_id, device)
        y = torch.tensor([it["labels"][r] for r in rows], device=device)
        loss = F.cross_entropy(head(H, emb_w[toks]), y)
        (loss / a.accum).backward()
        run_ce += float(loss.detach()) * len(y); run_n += len(y)
        if n_ctx % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if n_ctx % 500 == 0:
            print(f"[train] contexts {n_ctx}/{len(order)} ce {run_ce / max(1, run_n):.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
            run_ce, run_n = 0.0, 0
        if n_ctx % a.eval_every == 0 or n_ctx == len(order):
            ev = evaluate(model, head, emb_w, val_eval, pad_id, device) | {"contexts": n_ctx, "wall": time.time() - t0}
            log["evals"].append(ev)
            print(f"[eval {n_ctx}] ce {ev['ce']:.4f} acc {ev['acc']:.3f} | by position " +
                  " ".join(f"{k} {v:.3f}" for k, v in ev["ce_by_position"].items()) +
                  f" | AUROC in-mask-region: first token {ev['auroc_mask_region_first_token']:.3f}, mid {ev['auroc_mask_region_mid_sentence']:.3f}", flush=True)
            if ev["ce"] < best:
                best = ev["ce"]; binhead.save(Path(a.out) / "binhead.pt", head, meta | {"best_eval": ev})
            write_json(Path(a.out) / "train_log.json", log)
    print(f"done ({time.time() - t0:.0f}s); best validation ce {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
