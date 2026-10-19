"""Does bounding the reference model's context restore the key's influence late in a text?  (reference-only pass, no generation)

For existing clean and marked texts, the reference model scores every position twice (keyed / neutral context) while
seeing either the whole text so far ("full") or only the last W..1.5W tokens (blocks of S = W/2 positions share one
window, exactly as a windowed sampler would re-prefill).  Per position we record
  cap_fixed : Var[clip(d_W)] under the FULL-context neutral distribution -- how far the key moves the windowed
              reference's preferences among the tokens that are actually plausible there (sampling distribution held fixed);
  cap_own   : the same variance under the windowed neutral distribution;
  h0        : windowed neutral entropy (the detector's gate);   d_real : contrast of the realised token.
If the late-position capacity under window W matches the full-context capacity at positions ~W, the decay along a text is a
context-length effect and a windowed push + windowed detector would keep marking at the opening rate.
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmworld.config import load_config  # noqa: E402
from wmworld.posstats import auroc, gate_weights, weighted_t  # noqa: E402
from wmworld.reference import build_reference  # noqa: E402
from wmworld.scoring import LikelihoodScorer  # noqa: E402
from wmworld.stages import collect_run_texts, load_key_records  # noqa: E402

WINDOWS = [32, 64, 128, 256]
CLIP = 0.5
BINS = [0, 32, 64, 128, 256, 512, 1024, 4096]
SETS = [("conf_cross", "confirmation", ["base", "wm_c4"], 100), ("long_story", "len10", ["base", "wm_c3"], 50)]

cfg = load_config("configs/pilot.yaml")
ref = build_reference(cfg)
scorer = LikelihoodScorer(ref, max_input_tokens=ref.max_input_tokens)
dev = ref.device


@torch.no_grad()
def full_lp(ctx, toks):
    ids = torch.tensor([ctx + toks[:-1]], device=dev)
    return torch.log_softmax(ref.model(input_ids=ids, use_cache=False, logits_to_keep=len(toks)).logits[0].float(), dim=-1)


@torch.no_grad()
def block_lp(ctx, toks, blocks):
    """blocks: same-shape (a, s, e) triples -> [B, e-s, V] log-probs of positions s..e-1 seeing tokens a..position-1."""
    k = blocks[0][2] - blocks[0][1]
    ids = torch.tensor([ctx + toks[a:e - 1] for a, s, e in blocks], device=dev)
    return torch.log_softmax(ref.model(input_ids=ids, use_cache=False, logits_to_keep=k).logits.float(), dim=-1)


def metrics(lpk, lp0, p_fixed, tok):
    d = lpk - lp0
    c = d.clamp(-CLIP, CLIP)
    mu_f = (p_fixed * c).sum(-1)
    cap_f = (p_fixed * c * c).sum(-1) - mu_f * mu_f
    p0 = lp0.exp()
    mu_o = (p0 * c).sum(-1)
    cap_o = (p0 * c * c).sum(-1) - mu_o * mu_o
    h0 = -(p0 * lp0).sum(-1)
    dr = d.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return [x.cpu().numpy().astype(np.float32) for x in (cap_f.clamp_min(0), cap_o.clamp_min(0), h0, dr)]


def score_text(ctxk, ctx0, toks):
    n = len(toks)
    tok = torch.tensor(toks, device=dev)
    lp0_full = full_lp(ctx0, toks)
    p_full = lp0_full.exp()
    out = {"full": metrics(full_lp(ctxk, toks), lp0_full, p_full, tok)}
    del lp0_full
    for W in WINDOWS:
        S = W // 2
        res = [np.zeros(n, dtype=np.float32) for _ in range(4)]
        groups = defaultdict(list)
        s = 0
        while s < n:
            e = min(s + S, n); a = max(0, s - W)
            groups[(e - 1 - a, e - s)].append((a, s, e)); s = e
        for blocks in groups.values():
            for b0 in range(0, len(blocks), 16):
                bl = blocks[b0:b0 + 16]
                lpk, lp0 = block_lp(ctxk, toks, bl), block_lp(ctx0, toks, bl)
                for r, (a, s, e) in enumerate(bl):
                    for arr, v in zip(res, metrics(lpk[r], lp0[r], p_full[s:e], tok[s:e])):
                        arr[s:e] = v
                del lpk, lp0
        out[str(W)] = res
    return out


t0 = time.time()
records, store = [], defaultdict(list)
for run, key_set, arms, n_prompts in SETS:
    keys = {k.key_id: k.to_keyspec() for k in load_key_records(cfg, key_set)}
    gens, _ = collect_run_texts(Path("runs") / run)
    by = defaultdict(dict)
    for g in gens:
        if g.get("status", "ok") == "ok" and g["arm"] in arms and g.get("text"):
            by[g["prompt_id"]][g["arm"]] = g
    pids = sorted((p for p, v in by.items() if all(a in v for a in arms)), key=lambda p: hashlib.sha256(p.encode()).hexdigest())[:n_prompts]
    for i, pid in enumerate(pids):
        for arm in arms:
            g = by[pid][arm]
            ks = keys[g["assigned_key_id"]]
            ctxk, ctx0 = scorer.context_ids(ks, neutral=False), scorer.context_ids(ks, neutral=True)
            toks = scorer.tokenize(g["text"])
            if len(toks) < 64 or len(ctxk) + len(toks) > ref.max_input_tokens:
                continue
            res = score_text(ctxk, ctx0, toks)
            records.append({"run": run, "arm": arm, "marked": arm != "base", "n": len(toks), "domain": g.get("domain")})
            for w, arrs in res.items():
                store[w].append(arrs)
        if (i + 1) % 10 == 0:
            print(f"[{run}] {i + 1}/{len(pids)} prompts ({time.time() - t0:.0f}s)", flush=True)

dest = Path("runs/window_capacity")
dest.mkdir(exist_ok=True)
offsets = np.concatenate([[0], np.cumsum([r["n"] for r in records])])
np.savez_compressed(dest / "positions.npz", offsets=offsets, meta=np.array(json.dumps(records)),
                    **{f"{w}_{nm}": np.concatenate([a[j] for a in arrs]) for w, arrs in store.items() for j, nm in enumerate(("cap_fixed", "cap_own", "h0", "d_real"))})

# ------------------------------------------------------------------ summary ------------------------------------------------------------------ #
order = ["full"] + [str(w) for w in WINDOWS]
labels = [f"{a}-{b}" for a, b in zip(BINS[:-1], BINS[1:])]
summary = {}
for run, *_ in SETS:
    idx_c = [i for i, r in enumerate(records) if r["run"] == run and not r["marked"]]
    idx_m = [i for i, r in enumerate(records) if r["run"] == run and r["marked"]]
    gate_full = [gate_weights(store["full"][i][2], 0.2, 1.0) for i in range(len(records))]
    print(f"\n{'=' * 110}\n{run}: {len(idx_c)} clean + {len(idx_m)} marked texts; median {int(np.median([records[i]['n'] for i in idx_c]))} reference tokens (clean)")
    summary[run] = {}
    for title, j in (("A. capacity Var[clip d] under the FIXED full-context distribution (clean texts, gated)", 0),
                     ("B. capacity under the window's own neutral distribution (clean texts, gated)", 1)):
        print(f"\n{title}\n{'context':>8s} " + " ".join(f"{l:>10s}" for l in labels) + f" {'whole text':>11s}")
        for w in order:
            num, den = np.zeros(len(labels)), np.zeros(len(labels))
            for i in idx_c:
                x, g = store[w][i][j], gate_full[i]
                pos = np.arange(x.size)
                for b, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
                    m = (pos >= lo) & (pos < hi)
                    num[b] += float((g[m] * x[m]).sum()); den[b] += float(g[m].sum())
            row = num / np.maximum(den, 1e-9)
            summary[run][f"{'fixed' if j == 0 else 'own'}_{w}"] = {"by_position": [float(v) if d > 0 else None for v, d in zip(row, den)], "whole": float(num.sum() / den.sum())}
            print(f"{w:>8s} " + " ".join(f"{v:10.4f}" if d > 50 else f"{'-':>10s}" for v, d in zip(row, den)) + f" {num.sum() / den.sum():11.4f}")
    print(f"\nC. existing (full-context-made) mark seen through each context: gated clipped evidence per token, marked minus clean\n{'context':>8s} "
          + " ".join(f"{l:>10s}" for l in labels) + f" {'T marked':>9s} {'T clean':>8s} {'AUROC':>6s}")
    for w in order:
        prof = {}
        for name, idx in (("m", idx_m), ("c", idx_c)):
            num, den = np.zeros(len(labels)), np.zeros(len(labels))
            for i in idx:
                h0, d = store[w][i][2], store[w][i][3]
                g, ev = gate_weights(h0, 0.2, 1.0), np.clip(d, -CLIP, CLIP)
                pos = np.arange(d.size)
                for b, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
                    m = (pos >= lo) & (pos < hi)
                    num[b] += float((g[m] * ev[m]).sum()); den[b] += float(g[m].sum())
            prof[name] = (num / np.maximum(den, 1e-9), den)
        T = {name: [weighted_t(np.clip(store[w][i][3], -CLIP, CLIP), gate_weights(store[w][i][2], 0.2, 1.0)) for i in idx] for name, idx in (("m", idx_m), ("c", idx_c))}
        diff = prof["m"][0] - prof["c"][0]
        au = auroc(T["m"], T["c"])
        summary[run][f"seen_{w}"] = {"marked_minus_clean": [float(v) for v in diff], "T_marked": float(np.mean([t for t in T["m"] if t is not None])),
                                     "T_clean": float(np.mean([t for t in T["c"] if t is not None])), "auroc": au}
        print(f"{w:>8s} " + " ".join(f"{v:+10.4f}" if d > 50 else f"{'-':>10s}" for v, d in zip(diff, prof['m'][1]))
              + f" {summary[run][f'seen_{w}']['T_marked']:9.2f} {summary[run][f'seen_{w}']['T_clean']:8.2f} {au:6.3f}")
json.dump(summary, open(dest / "summary.json", "w"), indent=1)
print(f"\nwrote {dest}/summary.json and positions.npz ({time.time() - t0:.0f}s)")
