"""Step-by-step trace of push + margin on one benchmark prompt, for illustration (CPU, fp32, fixed seeds; the first
ob_wikipedia test prompt, not selected).  Uses the real NucleusSteerProcessor and unit_candidates:
  1. a plain first sentence -> its region -> the mask for sentence 2
  2. one candidate for sentence 2 written token by token with the push: entropy, gate, the top candidates with their
     probability before/after, the depth of "sentence so far + token", and the token sampled
  3. one block of 4 candidates with the push and, same seed, without it: regions, depths, usable, which is kept
Writes runs/push_trace.json."""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from wmworld import semwm
import steered_semstamp as S

torch.set_grad_enabled(False)
M = json.load(open("assets/ss_models.json"))
lm = S.LM(M["llama3b"]["path"], "fp32", device="cpu")
emb = semwm.Embedder(S.SEMSTAMP_SBERT, device="cpu", batch_size=64)
key = semwm.Key.shipped(); planes = semwm.Hyperplanes.for_key(key, emb.dim)
V = lm.model.get_output_embeddings().weight.shape[0]
boundary = semwm.boundary_token_mask(lm.tokenizer, lm.eos_ids + [lm.pad_id], "cpu", V)
T, TOP_P = 0.8, 0.95
sampling = SimpleNamespace(temperature=T, top_p=TOP_P)
warp = LogitsProcessorList([TemperatureLogitsWarper(T), TopKLogitsWarper(50), TopPLogitsWarper(TOP_P)])   # HF's sampler
tok = lambda i: lm.tokenizer.decode([i])
out = {}

p = next(json.loads(l) for l in open("data/ob_bench/prompts.jsonl") if '"ob_wikipedia"' in l and '"test"' in l)
ctx = lm.chat_prefix_ids(S.GENERATOR_SYSTEM, p["prompt"])
out["prompt_id"], out["excerpt"] = p["prompt_id"], p["excerpt"]

# 1. plain first sentence
torch.manual_seed(7)
row = S._generate(lm, ctx, sampling, 64, 2, None, semwm.SentenceEndBatch(lm.tokenizer, len(ctx), lm.eos_ids), 1)[0]
raw, _ = S.clean_ids(lm, row)
first_ids, first_text, _ = semwm.trim_candidate(lm.tokenizer, raw, [])
q1 = planes.projections(emb.encode([first_text]))[0]
b1 = planes.bin_of(q1); mask = tuple(int(b) for b in semwm.valid_mask(key, b1, semwm.LMBD))
out["first"] = {"text": first_text, "proj": [round(float(x), 4) for x in q1], "region": int(b1), "mask_for_next": list(mask)}
print("FIRST:", repr(first_text), "proj", np.round(q1, 4), "region", b1, "-> mask", mask, flush=True)

prefix = ctx + first_ids
mk = lambda: semwm.NucleusSteerProcessor(planes, emb, lm.tokenizer, len(prefix), mask[0], 1.0, boundary, T, TOP_P, mode="marginal",
                                         stat="seg", topk=20, clip_sd=1.0, target_depth=0.05, rep_n=4, rep_L0=len(ctx),
                                         eos_ids=lm.eos_ids + [lm.pad_id], target_bins=mask)

# 2. one pushed candidate, token by token
proc = mk(); gen = torch.Generator().manual_seed(11)
ids, steps = list(prefix), []
for step in range(48):
    s = lm.model(torch.tensor([ids])).logits[0, -1].float()
    p_raw = torch.softmax(s, -1); H = float(-(p_raw * torch.log(p_raw + 1e-30)).sum())
    g = semwm.entropy_gate(H)
    tilted = proc(torch.tensor([ids]), s[None].clone())[0]
    p0 = torch.softmax(warp(torch.tensor([ids]), s[None].clone())[0], -1)
    p1 = torch.softmax(warp(torch.tensor([ids]), tilted[None].clone())[0], -1)
    rec = {"step": step, "entropy": round(H, 3), "gate": round(g, 3), "unit_so_far": lm.decode(ids[len(prefix):])}
    if g > 0:
        nuc = proc._nucleus(s); lex = nuc[~boundary[nuc]]
        order = torch.argsort(torch.softmax(s[lex] / T, -1), descending=True)[:20]
        push = lex[order]
        strings = proc._unit_texts(ids[len(prefix):], [int(v) for v in push])
        depth = proc._score(emb.encode(strings))
        cands = sorted(zip(push.tolist(), depth.tolist()), key=lambda x: -float(p0[x[0]]))
        rec["candidates"] = [{"token": tok(v), "p_before": round(float(p0[v]), 4), "p_after": round(float(p1[v]), 4),
                              "depth_if_chosen": round(d, 4), "logit_change": round(float(tilted[v] - s[v]), 3)} for v, d in cands[:6]]
        rec["n_candidates_scored"] = len(cands)
    nxt = int(torch.multinomial(p1, 1, generator=gen))
    rec["sampled"] = tok(nxt); rec["sampled_p_before"] = round(float(p0[nxt]), 4); rec["sampled_p_after"] = round(float(p1[nxt]), 4)
    steps.append(rec); ids.append(nxt)
    print(f"step {step:2d}  H {H:5.2f} gate {g:4.2f}  sampled {tok(nxt)!r:14s} p {float(p0[nxt]):.3f}->{float(p1[nxt]):.3f}", flush=True)
    if nxt in lm.eos_ids or semwm.first_boundary(lm.decode(ids[len(prefix):])) is not None:
        break
unit_ids, unit_text, _ = semwm.trim_candidate(lm.tokenizer, ids[len(prefix):], lm.eos_ids)
q2 = planes.projections(emb.encode([unit_text]))[0]
out["trace"] = {"steps": steps, "sentence": unit_text, "proj": [round(float(x), 4) for x in q2], "region": int(planes.bin_of(q2)),
                "depth_in_mask": round(float(semwm.signed_distance(q2, mask, planes)), 4), "clear_of_margin": bool(np.min(np.abs(q2)) >= 0.02)}
print("TRACED SENTENCE:", repr(unit_text), out["trace"]["region"], out["trace"]["depth_in_mask"], flush=True)

# 3. one block of 4, with and without the push (same seed)
a = SimpleNamespace(unit="sent", sent_max_tokens=64, chunk_max_tokens=48, eot_at_boundary=False, W=16)
for name, procs in (("with_push", [mk()]), ("without_push", None)):
    torch.manual_seed(1234)
    cands, _ = S.unit_candidates(lm, prefix, [], sampling, 4, a, 200, 0, len(first_ids), procs)
    P = planes.projections(emb.encode([c["text"] for c in cands]))
    rows = []
    for c, q in zip(cands, P):
        d = float(semwm.signed_distance(q, mask, planes))
        rows.append({"text": c["text"], "region": int(planes.bin_of(q)), "depth": round(d, 4), "usable": bool(d > 0 and np.min(np.abs(q)) >= 0.02)})
    use = [i for i, r in enumerate(rows) if r["usable"]]
    kept = max(use, key=lambda i: rows[i]["depth"]) if use else int(np.argmax([r["depth"] for r in rows]))
    out[name] = {"candidates": rows, "kept": kept, "stop_after_block": bool(use)}
    print(name, [(r["region"], r["depth"], r["usable"]) for r in rows], "kept", kept, flush=True)
json.dump(out, open("runs/push_trace.json", "w"), indent=1, ensure_ascii=False)
print("wrote runs/push_trace.json")
