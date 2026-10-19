"""Amendment 15: distil the true-shift push into a small head (wmworld/pushhead.py), then measure how faithfully it
reproduces the push on held-out positions.

build  positions = every token position of 2 corpus candidates per context (runs/ss_corpus: training prompts; the
       validation set from runs/ss_corpus_val); at each position the push's own computation (semwm.NucleusSteerProcessor,
       mode "marginal", target = the unit's mask, top 20 lexical nucleus tokens, 20 encoder calls) gives the targets z_v;
       positions where the push is inactive (entropy gate 0, or fewer than 2 lexical nucleus tokens) are skipped
train  MSE between the head's standardised outputs and z_v (same weighting and clip as the push), AdamW, epochs
eval   on validation positions: R^2 of z, mean Spearman correlation within each position's 20 tokens, agreement on the
       top-scored token, sign agreement; also for the push-inactive-but-flat case (all gains equal: z = 0)
Outputs: runs/ss_pushhead/{train,val}.pt, runs/ss_pushhead/v1/pushhead.pt, runs/ss_pushhead/v1/report.json
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

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ss_rl_mask", ROOT / "scripts" / "ss_rl_mask.py")
rl = importlib.util.module_from_spec(spec); sys.modules["ss_rl_mask"] = rl; spec.loader.exec_module(rl)
ss = rl.ss

from wmworld import pushhead, semwm  # noqa: E402
from wmworld.records import write_json  # noqa: E402

TOPK = 20


def corpus_items(run: str, prompts, tok, n_ctx: int, seed: int) -> list[dict]:
    items = rl.load_contexts(run, prompts, tok)
    import glob
    corpus = {}
    for f in sorted(glob.glob(f"{run}/cands.shard*.jsonl")):
        for ln in open(f):
            r = json.loads(ln); corpus[r["prompt_id"]] = r
    rnd = random.Random(seed)
    pick = rnd.sample(items, min(n_ctx, len(items)))
    for it in pick:
        u = next(u for u in corpus[it["pid"]]["units"] if u["unit"] == it["unit"])
        c = [x["ids"] for x in u["cands"] if x["ids"]]
        it["cands"] = rnd.sample(c, min(2, len(c)))
    return pick


@torch.no_grad()
def build(lm, emb, planes, boundary, items, sampling) -> dict:
    procs = {}
    H, TOP, W, Z, V, GATE, MID, STATE, GAINS = [], [], [], [], [], [], [], [], []
    for n_it, it in enumerate(items):
        mask = it["mask"]
        if it["prev_bin"] not in procs:
            procs[it["prev_bin"]] = semwm.NucleusSteerProcessor(planes, emb, lm.tokenizer, 0, mask[0], 1.0, boundary, sampling.temperature,
                                                               sampling.top_p, mode="marginal", stat="seg", topk=TOPK, clip_sd=1.0,
                                                               target_depth=0.05, eos_ids=lm.eos_ids + [lm.pad_id], target_bins=mask)
        proc = procs[it["prev_bin"]]
        prefix, cands = it["prefix"], it["cands"]
        R = len(cands)
        cache = lm.model.model(input_ids=torch.tensor([prefix[:-1]], device=lm.device), use_cache=True).past_key_values
        cache.batch_repeat_interleave(R)
        Lc = max(len(c) for c in cands)
        inp = torch.full((R, Lc), lm.pad_id, dtype=torch.long, device=lm.device)
        att = torch.zeros((R, len(prefix) - 1 + Lc), dtype=torch.long, device=lm.device); att[:, :len(prefix) - 1] = 1
        for r, c in enumerate(cands):
            seq = [prefix[-1]] + c[:-1]
            inp[r, :len(seq)] = torch.tensor(seq, device=lm.device); att[r, len(prefix) - 1:len(prefix) - 1 + len(seq)] = 1
        Hs = lm.model.model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state
        logits = lm.model.lm_head(Hs).float()
        jobs, strings, cur_strings = [], [], []
        for r, c in enumerate(cands):
            for t in range(len(c)):
                s = logits[r, t]
                if t < 2:
                    s = s.clone(); s[torch.tensor(lm.eos_ids, device=s.device)] = float("-inf")
                p_raw = torch.softmax(s, -1)
                g = semwm.entropy_gate(float(-(p_raw * torch.log(p_raw + 1e-30)).sum()), 0.2, 2.0)
                if g <= 0:
                    continue
                nuc = proc._nucleus(s)
                lex = nuc[~boundary[nuc]]
                if lex.numel() < 2:
                    continue
                p_lex = torch.softmax(s[lex] / sampling.temperature, -1)
                order = torch.argsort(p_lex, descending=True)[:TOPK]
                push = lex[order]
                unit = c[:t]
                strs = proc._unit_texts(unit, [int(v) for v in push.tolist()])
                jobs.append((r, t, push, p_lex[order], g, len(strings), len(strs), len(cur_strings) if unit else -1, len(unit)))
                strings += strs
                if unit:
                    cur_strings.append(lm.tokenizer.decode(unit, skip_special_tokens=True).strip() or ".")
        if not jobs:
            continue
        gains_all = proc._score(emb.encode(strings))
        cur_P = planes.projections(emb.encode(cur_strings)) if cur_strings else None
        for r, t, push, w, g, s0, ns, ci, nt in jobs:
            gains = torch.tensor(gains_all[s0:s0 + ns], dtype=torch.float32)
            k = len(push)
            ids = torch.full((TOPK,), -1, dtype=torch.long); ids[:k] = push.cpu()
            ww = torch.zeros(TOPK); ww[:k] = w.cpu()
            gg = torch.zeros(TOPK); gg[:k] = gains
            valid = torch.zeros(TOPK); valid[:k] = 1
            z = pushhead.standardise(gg[None], ww[None], valid[None])[0]
            H.append(Hs[r, t].to(torch.bfloat16).cpu()); TOP.append(ids); W.append(ww); Z.append(z); V.append(valid); GAINS.append(gg)
            GATE.append(g); MID.append(it["prev_bin"])
            STATE.append(torch.tensor(pushhead.state_features(cur_P[ci] if ci >= 0 else None, mask, planes, nt)))
        if (n_it + 1) % 100 == 0:
            print(f"  built {n_it + 1}/{len(items)} contexts, {len(Z)} positions", flush=True)
    return {"h": torch.stack(H), "top": torch.stack(TOP), "w": torch.stack(W), "z": torch.stack(Z), "valid": torch.stack(V),
            "gains": torch.stack(GAINS), "gate": torch.tensor(GATE), "mask_id": torch.tensor(MID), "state": torch.stack(STATE)}


def predict(head, d, emb_w, idx, device):
    top = d["top"][idx].to(device)
    e = emb_w[top.clamp(min=0)]
    raw = head(d["h"][idx].to(device), e, d["mask_id"][idx].to(device), d["state"][idx].to(device))
    return pushhead.standardise(raw, d["w"][idx].to(device), d["valid"][idx].to(device))


@torch.no_grad()
def evaluate(head, d, emb_w, device) -> dict:
    head.eval()
    P = torch.cat([predict(head, d, emb_w, torch.arange(i, min(i + 512, len(d["z"]))), device).cpu() for i in range(0, len(d["z"]), 512)])
    head.train()
    Z, Vv = d["z"], d["valid"].bool()
    active = (Z.abs().sum(-1) > 0)
    r2 = 1 - ((P - Z)[Vv & active[:, None]] ** 2).sum() / ((Z[Vv & active[:, None]] - Z[Vv & active[:, None]].mean()) ** 2).sum()
    rho, top1, sign = [], [], []
    for i in torch.nonzero(active).flatten().tolist():
        k = int(Vv[i].sum())
        if k < 3:
            continue
        a, b = P[i, :k].numpy(), Z[i, :k].numpy()
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        if ra.std() > 0 and rb.std() > 0:
            rho.append(float(np.corrcoef(ra, rb)[0, 1]))
        top1.append(int(a.argmax() == b.argmax()))
        nz = b != 0
        sign.append(float((np.sign(a[nz]) == np.sign(b[nz])).mean()) if nz.any() else np.nan)
    flat = ~active
    return {"positions": int(len(Z)), "active_positions": int(active.sum()), "r2_z": float(r2), "spearman_within_step": float(np.mean(rho)),
            "top1_agreement": float(np.mean(top1)), "sign_agreement": float(np.nanmean(sign)),
            "flat_positions_mean_abs_pred": float(P[flat].abs().mean()) if flat.any() else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-contexts", type=int, default=1500)
    ap.add_argument("--val-contexts", type=int, default=300)
    ap.add_argument("--out", default="runs/ss_pushhead")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    random.seed(0); torch.manual_seed(0)
    m = json.loads(Path("assets/ss_models.json").read_text())["llama3b"]
    lm = ss.LM(m["path"], "bf16", device="cuda:0")
    for p_ in lm.model.parameters():
        p_.requires_grad_(False)
    emb = semwm.Embedder(ss.SEMSTAMP_SBERT, device="cuda:0", batch_size=256)
    planes = semwm.Hyperplanes.for_key(semwm.Key.shipped(), emb.dim)
    boundary = semwm.boundary_token_mask(lm.tokenizer, lm.eos_ids + [lm.pad_id], lm.device)
    sampling = type("S", (), {"temperature": 0.8, "top_p": 0.95})()
    prompts = {r["prompt_id"]: r["prompt"] for r in map(json.loads, open("data/ss_train/prompts.jsonl"))}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, run, n in (("train", "runs/ss_corpus", a.train_contexts), ("val", "runs/ss_corpus_val", a.val_contexts)):
        f = out / f"{split}.pt"
        if f.exists():
            data[split] = torch.load(f)
        else:
            print(f"building {split} from {run} ({n} contexts) ({time.time() - t0:.0f}s)", flush=True)
            data[split] = build(lm, emb, planes, boundary, corpus_items(run, prompts, lm.tokenizer, n, 7 if split == "train" else 8), sampling)
            torch.save(data[split], f)
        d = data[split]
        print(f"{split}: {len(d['z'])} positions, push active (non-flat) in {int((d['z'].abs().sum(-1) > 0).sum())} ({time.time() - t0:.0f}s)", flush=True)
    emb_w = lm.model.get_input_embeddings().weight
    head = pushhead.PushHead(lm.model.config.hidden_size).to(lm.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(data["train"]["z"]); steps = a.epochs * math.ceil(n / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    report = {"args": vars(a), "evals": []}
    best = -1e9
    ev = evaluate(head, data["val"], emb_w, lm.device) | {"epoch": 0}; report["evals"].append(ev)
    print(f"[eval 0] {json.dumps(ev)}", flush=True)
    for epoch in range(1, a.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            pred = predict(head, data["train"], emb_w, idx, lm.device)
            z, v = data["train"]["z"][idx].to(lm.device), data["train"]["valid"][idx].to(lm.device)
            loss = (((pred - z) ** 2) * v).sum() / v.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss) * len(idx)
        ev = evaluate(head, data["val"], emb_w, lm.device) | {"epoch": epoch, "train_mse": tot / n, "wall": time.time() - t0}
        report["evals"].append(ev)
        print(f"[eval {epoch}] train mse {tot / n:.4f} | val R2 {ev['r2_z']:.3f} spearman {ev['spearman_within_step']:.3f} top1 {ev['top1_agreement']:.3f} "
              f"sign {ev['sign_agreement']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if ev["spearman_within_step"] > best:
            best = ev["spearman_within_step"]
            pushhead.save(out / "v1" / "pushhead.pt", head, {"key": semwm.Key.shipped().label(), "topk": TOPK, "best_eval": ev})
        write_json(out / "v1" / "report.json", report)
    print(f"done ({time.time() - t0:.0f}s); best within-step Spearman {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
