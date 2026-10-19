"""Bin head v2 (amendment 12): predict the finished sentence's 3 hyperplane projections (Gaussian per plane).

Same data and pairs as scripts/ss_train_binhead.py (frozen 3B, forward only; every position t of every corpus
candidate, input = final hidden state h_t + embedding of the next token), but the target is the candidate's 3
projections (continuous) instead of its 9-class region.  Loss = Gaussian NLL per plane in standardised units.
Validation on the SAME 300 validation contexts as v1 (random.Random(1) sample):
  - per plane: R^2 of the predicted mean, sign accuracy, share of targets within +-1 predicted sd; by position
  - region cross-entropy of the derived region probabilities (comparable to v1's 0.8019; label prior 0.8134)
  - AUROC of log P(region in the unit's mask) for "the candidate ends in the mask's clear region", first token / mid
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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ss_train_binhead", ROOT / "scripts" / "ss_train_binhead.py")
v1 = importlib.util.module_from_spec(spec); sys.modules["ss_train_binhead"] = v1; spec.loader.exec_module(v1)

from wmworld import binhead, semwm  # noqa: E402
from wmworld.posstats import auroc  # noqa: E402
from wmworld.records import content_hash, write_json  # noqa: E402


def attach_projections(items: list[dict], runs: list[str]) -> None:
    """Add each candidate's 3 projections to the v1 items (same corpus, same candidate order and filtering)."""
    corpus = {}
    for run in runs:
        for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
            for ln in open(f):
                r = json.loads(ln); corpus[r["prompt_id"]] = r
    for it in items:
        u = next(u for u in corpus[it["pid"]]["units"] if u["unit"] == it["unit"])
        cd = [x for x in u["cands"] if x["ids"]]
        assert [x["ids"] for x in cd] == it["cands"]
        it["proj"] = [x["proj"] for x in cd]


@torch.no_grad()
def evaluate(model, head, emb_w, items, pad_id, device, margin) -> dict:
    head.eval()
    Y, MU, SD, REL, REG_CE, ROW_KEYS = [], [], [], [], [], []
    first_s, first_y, mid_s, mid_y = [], [], [], []
    for it in items:
        H, toks, rows, rel = v1.hidden_pairs(model, it, pad_id, device)
        mu, sd = head(H, emb_w[toks])
        y = torch.tensor([it["proj"][r] for r in rows], device=device, dtype=torch.float32)
        lab = torch.tensor([it["labels"][r] for r in rows], device=device)
        lpr = binhead.region_logprobs_from_gaussian(mu, sd, margin)
        REG_CE.append(-lpr.gather(-1, lab[:, None]).squeeze(-1).cpu().numpy())
        Y.append(y.cpu().numpy()); MU.append(mu.cpu().numpy()); SD.append(sd.cpu().numpy()); REL.append(rel)
        ml = torch.logsumexp(lpr[:, [int(b) for b in it["mask"]]], dim=-1).cpu().numpy()
        inreg = np.array([it["labels"][r] in it["mask"] for r in rows]); rows_a = np.array(rows)
        for r in range(len(it["cands"])):
            idx = np.flatnonzero(rows_a == r)
            first_s.append(ml[idx[0]]); first_y.append(inreg[idx[0]]); mid_s.append(ml[idx[len(idx) // 2]]); mid_y.append(inreg[idx[len(idx) // 2]])
    head.train()
    Y, MU, SD, REL, REG_CE = map(np.concatenate, (Y, MU, SD, REL, REG_CE))
    out = {"pairs": len(Y), "region_ce": float(REG_CE.mean()), "planes": {}, "by_position": {}}
    for j in range(Y.shape[1]):
        out["planes"][j] = {"r2": float(1 - ((Y[:, j] - MU[:, j]) ** 2).mean() / Y[:, j].var()),
                            "sign_acc": float((np.sign(Y[:, j]) == np.sign(MU[:, j])).mean()),
                            "within_1sd": float((np.abs(Y[:, j] - MU[:, j]) <= SD[:, j]).mean())}
    for k, (lo, hi) in zip(("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"), [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]):
        s = (REL >= lo) & (REL < hi)
        out["by_position"][k] = {"r2_mean_over_planes": float(np.mean([1 - ((Y[s, j] - MU[s, j]) ** 2).mean() / Y[s, j].var() for j in range(Y.shape[1])])),
                                 "region_ce": float(REG_CE[s].mean())}
    au = lambda s, y: auroc([x for x, t in zip(s, y) if t], [x for x, t in zip(s, y) if not t])
    out.update({"auroc_mask_region_first_token": au(first_s, first_y), "auroc_mask_region_mid_sentence": au(mid_s, mid_y)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", default=[])
    ap.add_argument("--val", action="append", default=[])
    ap.add_argument("--out", default="runs/ss_binhead/v2_proj")
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
    train, val = v1.load_items(a.train, prompts, tok, a.margin), v1.load_items(a.val, prompts, tok, a.margin)
    attach_projections(train, a.train); attach_projections(val, a.val)
    val_eval = random.Random(1).sample(val, min(a.eval_n, len(val)))            # the same subset as v1
    P = np.array([p for it in train for p in it["proj"]])
    scale = P.std(axis=0)
    print(f"train contexts {len(train)}, val eval {len(val_eval)}; projection sd per plane {np.round(scale, 4).tolist()}", flush=True)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(m["path"], dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb_w = model.get_input_embeddings().weight
    head = binhead.ProjHead(model.config.hidden_size, a.hidden).to(device)
    head.scale.copy_(torch.tensor(scale, dtype=torch.float32))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    total = int(math.ceil(len(train) * a.epochs / a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    meta = {"key": semwm.Key.shipped().label(), "margin": a.margin, "args": vars(a), "scale": scale.tolist(),
            "corpus_hash": content_hash({"train": sorted({it["pid"] for it in train}), "n": len(train)}, length=12)}
    log = {"meta": meta, "evals": []}
    order = [i for _ in range(int(math.ceil(a.epochs))) for i in random.Random(2).sample(range(len(train)), len(train))][: int(len(train) * a.epochs)]
    best, run_l, run_n = float("inf"), 0.0, 0
    for n_ctx, i in enumerate(order, 1):
        it = train[i]
        H, toks, rows, _ = v1.hidden_pairs(model, it, pad_id, device)
        y = torch.tensor([it["proj"][r] for r in rows], device=device, dtype=torch.float32)
        mu, sd = head(H, emb_w[toks])
        loss = binhead.gaussian_nll(mu, sd, y, head.scale).mean()
        (loss / a.accum).backward()
        run_l += float(loss.detach()) * len(y); run_n += len(y)
        if n_ctx % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if n_ctx % 500 == 0:
            print(f"[train] contexts {n_ctx}/{len(order)} gaussian nll {run_l / max(1, run_n):.4f} ({time.time() - t0:.0f}s)", flush=True)
            run_l, run_n = 0.0, 0
        if n_ctx % a.eval_every == 0 or n_ctx == len(order):
            ev = evaluate(model, head, emb_w, val_eval, pad_id, device, a.margin) | {"contexts": n_ctx, "wall": time.time() - t0}
            log["evals"].append(ev)
            pl = ev["planes"]
            print(f"[eval {n_ctx}] R2 per plane {[round(pl[j]['r2'], 3) for j in pl]} sign acc {[round(pl[j]['sign_acc'], 3) for j in pl]} "
                  f"within 1sd {[round(pl[j]['within_1sd'], 3) for j in pl]} | R2 by position " +
                  " ".join(f"{k} {v['r2_mean_over_planes']:.3f}" for k, v in ev["by_position"].items()) +
                  f" | region ce {ev['region_ce']:.4f} | AUROC in-mask-region first {ev['auroc_mask_region_first_token']:.3f} mid {ev['auroc_mask_region_mid_sentence']:.3f}", flush=True)
            if ev["region_ce"] < best:
                best = ev["region_ce"]; binhead.save_proj(Path(a.out) / "projhead.pt", head, meta | {"best_eval": ev})
            write_json(Path(a.out) / "train_log.json", log)
    print(f"done ({time.time() - t0:.0f}s); best region ce {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
