"""Generation harness for the semantic-watermark arms on Llama-3.2-3B-Instruct, optionally drafted by Llama-3.2-1B.

Arms (one shared key: SemStamp's shipped constants, see wmworld/semwm.py):
  base           plain sampling
  semstamp       SemStamp as shipped: sentence-level rejection until the sentence's LSH bin is in the mask seeded by
                 the previous sentence and it clears the margin; N candidates per call
  steered_sent   the same acceptance rule, but every token is pushed toward the target bin (steer, then reject rarely)
  steered_seg    one fixed target bin for the whole text, push only, no rejection; detected with the segment statistic
Drafting:
  none           the 3B does everything
  assist         HF assisted generation (3B target + 1B assistant, same bias processor); exact 3B distribution, batch 1
  loop           the candidate loop runs on the 1B (batched); the accepted sentence is speculatively verified by the
                 3B, re-checked against the bin/margin rule and re-drafted if it moved out
Records follow scripts/baselines.py so the judge, attack and detection scripts apply unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmworld import semwm  # noqa: E402
from wmworld.config import load_config  # noqa: E402
from wmworld.generation import GENERATOR_SYSTEM, ends_mid_sentence  # noqa: E402
from wmworld.records import GenerationRecord, JsonlWriter, content_hash, output_id, repair_jsonl, write_json  # noqa: E402
from wmworld.textnorm import normalize_text, seed_from, stable_hash, text_hash  # noqa: E402

DATE = "26 Jul 2024"                      # the Llama-3.2 template inserts today's date; freeze it
PAD_TOKEN = "<|finetune_right_pad_id|>"
SEMSTAMP_SBERT = "/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert/snapshots/a77dce8530e100ac1101981e4be390546c1cb282"


# --------------------------------------------------------------------------- #
# models and prompts
# --------------------------------------------------------------------------- #

class LM:
    def __init__(self, path: str, dtype: str, device: str = "cuda"):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        arch = (json.loads((Path(path) / "config.json").read_text()).get("architectures") or [""])[0]
        if arch == "Qwen3_5ForConditionalGeneration":                   # Qwen3.5 checkpoints are multimodal; load the text model
            from transformers import Qwen3_5ForCausalLM
            self.model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=dt).to(device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(device).eval()
        self.device = self.model.device
        vocab = self.tokenizer.get_vocab()
        eos = self.model.generation_config.eos_token_id
        eos = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None]
        for t in ("<|im_end|>", "<|eot_id|>"):                           # end of the assistant turn (Qwen / Llama)
            if t in vocab and vocab[t] not in eos:
                eos.append(int(vocab[t]))
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in eos:
            eos.append(int(self.tokenizer.eos_token_id))
        self.eos_ids = eos
        pad = next(t for t in (PAD_TOKEN, "<|fim_pad|>", self.tokenizer.pad_token) if t in vocab and vocab[t] not in eos)
        self.pad_id = int(vocab[pad])
        assert self.pad_id not in self.eos_ids and self.pad_id >= 0
        self.n_forward = 0
        self.model.register_forward_hook(lambda m, i, o: setattr(self, "n_forward", self.n_forward + 1))

    def chat_prefix_ids(self, system: str, user: str) -> list[int]:
        text = self.tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                  tokenize=False, add_generation_prompt=True, date_string=DATE, enable_thinking=False)
        return [int(t) for t in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def plain_prefix_ids(self, text: str) -> list[int]:
        """--plain (base models): the raw text with the tokenizer's BOS, no chat template."""
        return [int(t) for t in self.tokenizer(text)["input_ids"]]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(t) for t in ids], skip_special_tokens=True)


def semstamp_prompt(text: str, len_prompt: int = 32) -> str:
    """SemStamp's prompt (abehou/SemStamp sampling_utils.extract_prompt_from_text, verbatim logic): the first
    (sub)sentence of the first len_prompt whitespace words, ended by the first of '!.?' leaving > 3 words, else + '.'."""
    new_text = " ".join(text.split(" ")[:len_prompt])
    prompts = []
    for pu in "!.?":
        idx = new_text.find(pu)
        if idx != -1 and len(new_text[:idx + 1].split(" ")) > 3:
            prompts.append(new_text[:idx + 1])
    if not prompts:
        prompts.append(new_text + ".")
    return sorted(prompts, key=len)[0]


def prompts_for(cfg, which: str, limit_per_domain: int, source_run: str, ob_domains=None):
    """held_out: exactly the 150 prompts of runs/conf_cross (with their key assignment); null_calibration: a fixed
    hash-ordered subset per domain of the null partition."""
    if which in ("ob_bench", "sp_bench", "it_bench"):                  # benchmarks: data/<which>/prompts.jsonl, test split
        want = set(ob_domains) if ob_domains else None
        rows = [SimpleNamespace(**r) for r in map(json.loads, open(f"data/{which}/prompts.jsonl"))
                if r["split"] == "test" and (want is None or r["domain"] in want)]
        return sorted(rows, key=lambda r: r.prompt_id), {r.prompt_id: None for r in rows}, f"{which}_test"
    if which.startswith("ss_train_"):                                  # bin-adapter corpus: data/ss_train/prompts.jsonl
        split = which.split("_")[-1]
        rows = [SimpleNamespace(**r) for r in map(json.loads, open("data/ss_train/prompts.jsonl")) if r["split"] == split]
        return sorted(rows, key=lambda r: r.prompt_id), {r.prompt_id: None for r in rows}, f"ss_train_{split}"
    if which == "held_out":
        rows, want = semwm.held_out_prompts(cfg, source_run)
        return rows, {p.prompt_id: want[p.prompt_id].get("assigned_key_id") for p in rows}, "positive_test"
    from wmworld.data.prepare import load_prepared
    rows = load_prepared(cfg.path_of("data_prepared"), partition="null_calibration")
    by_dom = defaultdict(list)
    for r in sorted(rows, key=lambda r: stable_hash("ss_null", cfg.seed, r.prompt_id)):
        by_dom[r.domain].append(r)
    rows = [r for d in sorted(by_dom) for r in by_dom[d][:limit_per_domain]]
    return sorted(rows, key=lambda r: r.prompt_id), {r.prompt_id: None for r in rows}, "null_calibration"


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def _generate(lm: LM, inp_ids: list[int], sampling, max_new: int, min_new: int, procs=None, stop=None, n: int = 1, assistant: LM | None = None):
    kw = dict(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p, min_new_tokens=min_new, max_new_tokens=max_new,
              eos_token_id=lm.eos_ids, pad_token_id=lm.pad_id)
    if getattr(sampling, "hf_top_k", None) is not None:                  # --top-k (0 = off); else the model's generation config
        kw["top_k"] = sampling.hf_top_k
    if getattr(sampling, "repetition_penalty", None):
        kw["repetition_penalty"] = sampling.repetition_penalty
    if procs:
        kw["logits_processor"] = LogitsProcessorList(list(procs))
    if stop is not None:
        kw["stopping_criteria"] = StoppingCriteriaList([stop])
    if assistant is not None:
        kw["assistant_model"] = assistant.model
    elif n > 1:
        kw["num_return_sequences"] = n
    out = lm.model.generate(torch.tensor([inp_ids], device=lm.device), **kw)
    return [[int(t) for t in row[len(inp_ids):].tolist()] for row in out]


def clean_ids(lm: LM, ids: list[int]) -> tuple[list[int], bool]:
    """Tokens before the first EOS/pad, and whether an EOS was seen."""
    out = []
    for t in ids:
        if t in lm.eos_ids:
            return out, True
        if t == lm.pad_id:
            break
        out.append(t)
    return out, False


@torch.no_grad()
def gen_whole(target: LM, ctx: list[int], sampling, seed: int, min_new: int, max_new: int, proc=None, assistant: LM | None = None):
    torch.manual_seed(seed)
    row = _generate(target, ctx, sampling, max_new, min_new, [proc] if proc else None, None, 1, assistant)[0]
    ids, ended = clean_ids(target, row)
    return ids, {"n_tokens": len(ids), "ended": ended}


@torch.no_grad()
def gen_sentencewise(target: LM, drafter: LM | None, ctx: list[int], prompt: str, key: semwm.Key, planes: semwm.Hyperplanes,
                     emb: semwm.Embedder, etok, sampling, seed: int, min_new: int, max_new: int, a) -> tuple[list[int], dict]:
    """SemStamp's loop: sentence by sentence, accept when bin in mask and margin clears.  With steering the candidates
    are pushed toward a target bin of the mask; with --draft loop the candidates come from the drafter and the accepted
    one is speculatively verified by the target."""
    lmbd, margin = a.lmbd, a.margin
    warpers = [TemperatureLogitsWarper(sampling.temperature), TopPLogitsWarper(sampling.top_p)]
    gen = torch.Generator(device=target.device)
    proposer = drafter if (a.draft == "loop" and drafter is not None) else target
    assistant = drafter if a.draft == "assist" else None
    n_cand = 1 if a.draft == "assist" else a.n_cand
    prev_bin = planes.bin_of(planes.projections(emb.encode([prompt]))[0])
    mask = semwm.valid_mask(key, prev_bin, lmbd)
    ids: list[int] = []
    st = {"trials": 0, "trials_target": 0, "accepted": 0, "maxed_out": 0, "margin_rejects": 0, "bin_rejects": 0,
          "left_region_after_verify": 0, "n_capped": 0, "verify_kept": 0, "verify_draft_len": 0, "sentence_trace": []}
    ended = False
    call = 0
    while len(ids) < max_new and not ended:
        if a.eot_at_boundary and ids and len(ids) >= min_new:       # the model may end the text here (see gen_chain)
            torch.manual_seed(seed_from(seed, "end", call))
            t1 = _generate(target, ctx + ids, sampling, 1, 0)[0]
            st["end_checks"] = st.get("end_checks", 0) + 1
            if t1 and t1[0] in target.eos_ids:
                ended = True; st["ended_at_boundary"] = True
                break
        cur = 0
        accepted = None
        while accepted is None:
            target_bin = mask[call % len(mask)]
            L0 = len(ctx) + len(ids)
            proc = None
            if a.beta > 0 and a.arm == "steered_sent":
                proc = semwm.SteerBiasProcessor(etok, planes, emb, target.tokenizer, L0, target_bin, a.beta, a.k, margin,
                                                eos_ids=target.eos_ids, zero_ids=[target.pad_id])
            torch.manual_seed(seed_from(seed, "sent", call))
            call += 1
            stop = semwm.SentenceEndBatch(target.tokenizer, L0, target.eos_ids)
            rows = _generate(proposer, ctx + ids, sampling, min(a.sent_max_tokens, max_new - len(ids)), 1 if a.eot_at_boundary else 2,
                             [proc] if proc else None, stop, n_cand, assistant)
            st["trials"] += len(rows)
            cur += len(rows)
            best = None
            for row in rows:
                raw, saw_eos = clean_ids(proposer, row)
                cand, piece, found = semwm.trim_candidate(target.tokenizer, raw, [])
                if not cand:
                    continue
                if saw_eos and not found and len(ids) + len(cand) < min_new:
                    continue                                   # too short to end here
                if not found and not saw_eos:
                    st["n_capped"] += 1
                pr = planes.projections(emb.encode([piece]))[0]
                b, ok_m = planes.bin_of(pr), planes.margin_ok(pr, margin)
                if b in mask and ok_m:
                    best = (cand, piece, saw_eos and (not found or (a.eot_at_boundary and len(cand) == len(raw))), b, pr)
                    break
                st["margin_rejects" if (b in mask) else "bin_rejects"] += 1
                if best is None:
                    best = None
            if best is None:
                if cur >= a.max_trials:                        # SemStamp: give up and keep an unfiltered candidate
                    raw, saw_eos = clean_ids(proposer, rows[0])
                    cand, piece, found = semwm.trim_candidate(target.tokenizer, raw, [])
                    if not cand:
                        ended = True; break
                    pr = planes.projections(emb.encode([piece]))[0]
                    best = (cand, piece, saw_eos and (not found or (a.eot_at_boundary and len(cand) == len(raw))), planes.bin_of(pr), pr)
                    st["maxed_out"] += 1
                else:
                    continue
            cand, piece, ends, b, pr = best
            if proposer is not target:                         # speculative verification of the drafted sentence
                gen.manual_seed(seed_from(seed, "verify", call))
                kept, info = semwm.verify_draft(target.model, drafter.model, ctx + ids, cand, [proc] if proc else [], warpers, L0, gen)
                st["trials_target"] += 1; st["verify_kept"] += info["n_kept"]; st["verify_draft_len"] += info["n_draft"]
                kept2, piece2, found2 = semwm.trim_candidate(target.tokenizer, kept, target.eos_ids)
                pr2 = planes.projections(emb.encode([piece2]))[0] if piece2 else pr
                b2 = planes.bin_of(pr2)
                if kept2 and (b2 in mask and planes.margin_ok(pr2, margin)):
                    cand, piece, b, pr = kept2, piece2, b2, pr2
                    ends = any(t in target.eos_ids for t in kept) and not found2
                else:
                    st["left_region_after_verify"] += 1
                    if cur < a.max_trials:
                        continue                               # re-draft
                    cand, piece, b, pr = (kept2 or cand), (piece2 or piece), b2, pr2
            accepted = (cand, piece, ends, b, pr)
        cand, piece, ends, b, pr = accepted
        ids += cand
        st["accepted"] += 1
        st["sentence_trace"].append({"bin": int(b), "in_mask": bool(b in mask), "min_abs_proj": float(np.min(np.abs(pr))), "n_tok": len(cand)})
        prev_bin = b
        mask = semwm.valid_mask(key, prev_bin, lmbd)
        ended = ends
    st["n_tokens"] = len(ids)
    return ids, st



@torch.no_grad()
def gen_bestof_chunks(target: LM, ctx: list[int], key: semwm.Key, planes: semwm.Hyperplanes, emb: semwm.Embedder,
                      sampling, seed: int, min_new: int, max_new: int, a, make_proc=None) -> tuple[list[int], dict]:
    """Best-of-N selection with NO logit bias: for every W-word chunk draw N exact samples and keep the one that sits
    deepest in the fixed target bin (what the segment statistic rewards).  Each candidate is an exact sample of the
    model, so the only distortion is the selection itself; the cost is a bounded N x, never a rejection loop.
    ``make_proc(L0)``: optional nucleus steerer per chunk (arm E: mild push, then select).  Selection uses the same
    per-unit score the detector sums (``--steer-stat``: seg depth or target-direction projection)."""
    target_bin = semwm.fixed_bin(key, a.lmbd)
    u = semwm.target_direction(planes, planes.bin_signs(target_bin), np.ones(planes.d, dtype=bool))
    depth_of = (lambda t: semwm.chunk_depth(t, target_bin, planes, emb)) if a.steer_stat == "seg" else \
        (lambda t: float(emb.encode([t])[0].astype(np.float64) @ u))
    ids: list[int] = []
    chunk_texts: list[str] = []
    steer = defaultdict(float)
    st = {"chunks": 0, "candidates": 0, "n_short": 0, "depth_chosen": [], "depth_mean": []}
    ended, chunk = False, 0
    while len(ids) < max_new and not ended:
        L0 = len(ctx) + len(ids)
        torch.manual_seed(seed_from(seed, "chunk", chunk)); chunk += 1
        stop = semwm.WordCountStop(target.tokenizer, L0, target.eos_ids, a.W)
        proc = make_proc(L0) if make_proc else None
        rows = _generate(target, ctx + ids, sampling, min(a.chunk_max_tokens, max_new - len(ids)), 2, [proc] if proc else None, stop, a.best_of)
        if proc:
            for k2, v2 in proc.summary().items():
                steer[k2] += v2
        st["candidates"] += len(rows)
        cands = []
        for row in rows:
            raw, saw_eos = clean_ids(target, row)
            cid, text, complete = semwm.trim_to_words(target.tokenizer, raw, a.W)
            if not cid or not text:
                continue
            ends = saw_eos and not complete
            if ends and len(ids) + len(cid) < min_new:
                st["n_short"] += 1; continue
            cands.append((depth_of(text), cid, ends, text))
        if not cands:
            ended = True; break
        depth, cid, ends, text = max(cands, key=lambda c: c[0])
        chunk_texts.append(text)
        st["depth_chosen"].append(depth); st["depth_mean"].append(float(np.mean([c[0] for c in cands])))
        ids += cid; ended = ends; st["chunks"] += 1
    segs = semwm.split_segments(target.decode(ids), a.W)          # do the chosen chunks ARE the detector's units?
    full = [" ".join(normalize_text(t).split()) for t in chunk_texts]
    st["align_units"] = len(segs); st["align_ok"] = sum(1 for x, y in zip(full, segs) if x == y)
    if steer:
        n_ch = max(1, st["chunks"])
        st.update({k2: (v2 / n_ch if k2 not in ("steer_rep_guard_hits", "steer_embed_calls", "steer_strings", "steer_calls") else v2)
                   for k2, v2 in steer.items()})
    st.update({"n_tokens": len(ids), "ended": ended,
               "depth_chosen": float(np.mean(st["depth_chosen"])) if st["depth_chosen"] else None,
               "depth_mean": float(np.mean(st["depth_mean"])) if st["depth_mean"] else None})
    return ids, st

def nucleus_proc(a, target: LM, emb, etok, planes, key, boundary, sampling, L0: int, rep_L0: int, seg_W):
    return semwm.NucleusSteerProcessor(planes, emb, target.tokenizer, L0, semwm.fixed_bin(key, a.lmbd), a.beta, boundary,
                                       sampling.temperature, sampling.top_p, mode=a.steer, etok=etok, stat=a.steer_stat,
                                       topk=a.steer_topk, clip_sd=a.steer_clip, target_depth=a.steer_target, rep_n=a.rep_n,
                                       rep_L0=rep_L0, seg_W=seg_W, k=a.k, eos_ids=target.eos_ids + [target.pad_id])


# --------------------------------------------------------------------------- #
# bounded chain: best-of-N per unit (sentence or W-word segment), optional 1B drafts checked by the 3B
# --------------------------------------------------------------------------- #

def _unit_stop(lm: LM, unit_L0: int, a):
    return semwm.SentenceEndBatch(lm.tokenizer, unit_L0, lm.eos_ids) if a.unit == "sent" else semwm.WordCountStop(lm.tokenizer, unit_L0, lm.eos_ids, a.W)


def _trim_unit(tok, ids: list[int], a):
    return semwm.trim_candidate(tok, ids, []) if a.unit == "sent" else semwm.trim_to_words(tok, ids, a.W)


@torch.no_grad()
def unit_candidates(lm: LM, prefix: list[int], partial: list[int], sampling, n: int, a, room: int, min_new: int, done: int, procs=None):
    """n samples of the unit being written; ``partial`` = its tokens already fixed.  -> (candidates, tokens computed)."""
    cap = a.sent_max_tokens if a.unit == "sent" else a.chunk_max_tokens
    max_new = max(2, min(cap, room) - len(partial))
    # EOS is suppressed for a unit's first tokens (no empty units); with --eot-at-boundary only for the first one -- the end
    # check before the unit already decided it -- so a 1-token unit such as a closing ``` can be followed by end of turn
    rows = _generate(lm, prefix + partial, sampling, max_new, 1 if a.eot_at_boundary else 2, procs, _unit_stop(lm, len(prefix), a), n)
    spent = len(rows) * max(len(r) for r in rows)                        # a batch pays for its longest row
    out = []
    for r in rows:
        raw, saw_eos = clean_ids(lm, r)
        cid, text, found = _trim_unit(lm.tokenizer, partial + raw, a)
        if not cid or not text:
            continue
        ends = saw_eos and (not found or (a.eot_at_boundary and len(cid) == len(partial) + len(raw)))   # EOS right after the boundary
        if ends and done + len(cid) < min_new:
            continue
        out.append({"ids": cid, "text": text, "ends": ends, "found": bool(found)})
    return out, spent


@torch.no_grad()
def sampler_check(target: LM, prefix: list[int], cands: list[list[int]], sampling):
    """One teacher-forced 3B pass: for each candidate, the first token the 3B's own sampler could NOT have produced
    (outside its temperature + top-k + top-p set), or None.  Also returns the processed logits at that position."""
    L, T = len(prefix), max(len(c) for c in cands)
    seq = torch.full((len(cands), L + T), target.pad_id, dtype=torch.long, device=target.device)
    att = torch.zeros_like(seq)
    for b, c in enumerate(cands):
        seq[b, :L + len(c)] = torch.tensor(prefix + c, device=target.device); att[b, :L + len(c)] = 1
    logits = target.model(input_ids=seq, attention_mask=att, use_cache=False, logits_to_keep=T + 1).logits   # [B, T+1, V]
    top_k = int(getattr(target.model.generation_config, "top_k", 0) or 0)
    viol, rows = [], []
    for b, c in enumerate(cands):
        s = logits[b, : len(c)].float() / sampling.temperature                # position t predicts token c[t]
        pr = torch.softmax(s, dim=-1)
        pt = pr[torch.arange(len(c)), torch.tensor(c, device=pr.device)]
        higher = pr > pt[:, None]
        bad = (pr * higher).sum(-1) >= sampling.top_p
        if top_k:
            bad |= higher.sum(-1) >= top_k
        idx = torch.nonzero(bad).flatten()
        t = int(idx[0]) if len(idx) else None
        viol.append(t); rows.append(s[t] if t is not None else None)
    return viol, rows


def _sample_processed(s: torch.Tensor, sampling, top_k: int, gen: torch.Generator) -> int:
    pr = torch.softmax(s, dim=-1)
    sp, idx = torch.sort(pr, descending=True)
    keep = (torch.cumsum(sp, 0) - sp) < sampling.top_p
    if top_k:
        keep[top_k:] = False
    sp = sp * keep
    return int(idx[torch.multinomial(sp / sp.sum(), 1, generator=gen)].item())


@torch.no_grad()
def draft_verify(target: LM, drafter: LM, prefix: list[int], mask, planes, emb, sampling, a, room, min_new, done, st, gen):
    """1B drafts N units; the deepest ``verify_top`` in-mask drafts are checked by one 3B pass; the first that the 3B
    sampler could have produced wholly is kept.  Otherwise the deepest draft is cut at its first bad token, that token
    is re-drawn from the 3B, and the 1B re-drafts the rest (up to ``max_repairs``); last resort: the 3B finishes the unit.
    Every token of the result is one the 3B's sampler allows."""
    partial: list[int] = []
    rec = {"rounds": []}
    top_k = int(getattr(target.model.generation_config, "top_k", 0) or 0)
    for rnd in range(a.max_repairs + 1):
        cands, spent = unit_candidates(drafter, prefix, partial, sampling, a.best_of, a, room, min_new, done)
        st["d_gen"] += spent
        if not cands:
            break
        P = planes.projections(emb.encode([c["text"] for c in cands]))
        dep = np.array([semwm.signed_distance(q, mask, planes) for q in P])
        order = [int(i) for i in np.argsort(-dep)[: a.verify_top]]
        viol, rows = sampler_check(target, prefix, [cands[i]["ids"] for i in order], sampling)
        st["t_score"] += sum(len(cands[i]["ids"]) for i in order)
        rec["rounds"].append({"n": len(cands), "in_mask": float(np.mean(dep > 0)), "depth_top": float(dep[order[0]]),
                              "viol": [-1 if v is None else v for v in viol], "len": [len(cands[i]["ids"]) for i in order]})
        for j, i in enumerate(order):
            if viol[j] is None:
                rec.update({"valid": True, "round": rnd}); return cands[i], P[i], rec
        i0, t = order[0], viol[0]
        tok = _sample_processed(rows[0], sampling, top_k, gen)
        st["repairs"] += 1
        if tok in target.eos_ids:                                         # the 3B wants to stop here
            c = {"ids": cands[i0]["ids"][:t], "ends": True, "found": False}
            c["text"] = target.decode(c["ids"]).strip()
            if c["ids"] and c["text"]:
                q = planes.projections(emb.encode([c["text"]]))[0]
                rec.update({"valid": True, "round": rnd, "eos_repair": True}); return c, q, rec
        partial = cands[i0]["ids"][:t] + [tok]
    st["fallback"] += 1
    cands, spent = unit_candidates(target, prefix, partial, sampling, 1, a, room, min_new, done)
    st["t_gen"] += spent
    c = cands[0] if cands else {"ids": partial, "text": target.decode(partial).strip() or ".", "ends": True, "found": False}
    rec.update({"valid": True, "round": a.max_repairs + 1, "fallback": True})
    return c, planes.projections(emb.encode([c["text"]]))[0], rec


@torch.no_grad()
def draft_verify2(target: LM, drafter: LM, prefix: list[int], mask, planes, emb, sampling, dsampling, a, room, min_new, done, st, gen):
    """v2 of the 1B-draft path (fixes from the smoke diagnosis, scripts/ss_draft_diag.py):
      1. a draft that passes the 3B sampler check is kept only if it is IN the mask (in-mask drafts are checked first);
         an out-of-mask draft is kept only when no draft of the round reached the mask;
      2. repair = substitute the first bad token by a 3B sample and keep the rest of the draft (re-trim, re-check),
         instead of cutting and re-drafting two thirds of the unit; ``max_repairs`` re-checks per unit, shared;
      3. the 1B drafts with its own, tighter sampler (``dsampling``) so fewer of its tokens fall outside the 3B's set.
    Last resorts: re-draft from the longest verified prefix, then the 3B finishes the unit."""
    top_k = int(getattr(target.model.generation_config, "top_k", 0) or 0)
    eos = set(target.eos_ids)
    rec = {"rounds": [], "subs": 0}
    budget = a.max_repairs
    partial: list[int] = []
    best_valid = None                                                    # closest-to-mask valid unit seen (out of mask)

    def evaluate(ids_, ends_):
        text = target.decode(ids_).strip()
        if not ids_ or not text:
            return None
        q = planes.projections(emb.encode([text]))[0]
        return {"ids": ids_, "text": text, "ends": ends_, "found": True}, q, float(semwm.signed_distance(q, mask, planes))

    for rnd in range(2):                                                 # round 0 from scratch, round 1 re-draft from a verified prefix
        cands, spent = unit_candidates(drafter, prefix, partial, dsampling, a.best_of, a, room, min_new, done)
        st["d_gen"] += spent
        if not cands:
            break
        P = planes.projections(emb.encode([c["text"] for c in cands]))
        dep = np.array([semwm.signed_distance(q, mask, planes) for q in P])
        order = [int(i) for i in np.argsort(-dep)]
        inm = [i for i in order if dep[i] > 0]
        check = (inm or order)[: a.verify_top]
        viol, rows = sampler_check(target, prefix, [cands[i]["ids"] for i in check], sampling)
        st["t_score"] += sum(len(cands[i]["ids"]) for i in check)
        rec["rounds"].append({"n": len(cands), "in_mask": float(np.mean(dep > 0)), "n_in": len(inm), "depth_top": float(dep[order[0]]),
                              "viol": [-1 if v is None else v for v in viol], "len": [len(cands[i]["ids"]) for i in check]})
        for j, i in enumerate(check):                                    # fix 1: pass AND in mask
            if viol[j] is None and (dep[i] > 0 or not inm):
                rec.update({"path": "pass", "round": rnd}); return cands[i], P[i], rec
            if viol[j] is None and (best_valid is None or dep[i] > best_valid[2]):
                best_valid = (cands[i], P[i], float(dep[i]))
        # fix 2: token substitution on the checked drafts, deepest first, while the shared budget lasts
        for j, i in enumerate(check):
            if viol[j] is None:
                continue
            ids_, t, row, ends_ = list(cands[i]["ids"]), viol[j], rows[j], cands[i]["ends"]
            while budget > 0:
                tok = _sample_processed(row, sampling, top_k, gen)
                budget -= 1; rec["subs"] += 1; st["repairs"] += 1
                if tok in eos and done + t < min_new:
                    continue                                             # too early to end the text: draw again
                if tok in eos:
                    ids_, ends_, t = ids_[:t], True, None
                else:
                    ids_ = ids_[:t] + [tok] + ids_[t + 1:]
                    cid, _, found = _trim_unit(target.tokenizer, ids_, a)
                    if not found and cands[i]["found"]:                  # substitution broke the unit (boundary / word count)
                        partial = cid[:t + 1]; break
                    ids_ = cid
                    v2, r2 = sampler_check(target, prefix, [ids_], sampling)
                    st["t_score"] += len(ids_)
                    t, row = v2[0], r2[0]
                if t is None:
                    ev = evaluate(ids_, ends_)
                    if ev is not None and ev[2] > 0:
                        rec.update({"path": "sub", "round": rnd}); return ev[0], ev[1], rec
                    if ev is not None and (best_valid is None or ev[2] > best_valid[2]):
                        best_valid = ev
                    break
            else:
                partial = ids_[:t] if t is not None else ids_
            if budget <= 0:
                break
        if best_valid is not None and budget <= 0:
            break
        if not partial:                                                  # re-draft from the deepest draft's verified prefix
            partial = cands[check[0]]["ids"][: viol[0]] if viol[0] is not None else []
    if best_valid is not None:
        rec.update({"path": "valid_out"}); return best_valid[0], best_valid[1], rec
    st["fallback"] += 1
    cands, spent = unit_candidates(target, prefix, partial, sampling, 1, a, room, min_new, done)
    st["t_gen"] += spent
    c = cands[0] if cands else {"ids": partial, "text": target.decode(partial).strip() or ".", "ends": True, "found": False}
    rec.update({"path": "fallback"})
    return c, planes.projections(emb.encode([c["text"]]))[0], rec


# --------------------------------------------------------------------------- #
# bin adapters (amendment 11): sample units from the adapted 3B, optional SIR back to the model's own conditional
# --------------------------------------------------------------------------- #

def processed_logprobs(logits: torch.Tensor, temperature: float, top_p: float, top_k: int | None = None) -> torch.Tensor:
    """log-probabilities of the sampler the 3B actually uses (HF order: temperature, top-k if set, top-p), -inf outside."""
    s = logits.float() / temperature
    if top_k:
        kth = torch.topk(s, min(int(top_k), s.shape[-1]), dim=-1).values[..., -1:]
        s = s.masked_fill(s < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        srt, idx = torch.sort(s, descending=False, dim=-1)
        remove = srt.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
        remove[..., -1:] = False
        s = s.masked_fill(remove.scatter(-1, idx, remove), float("-inf"))
    return torch.log_softmax(s, dim=-1)


def _sampler_top_k(lm: LM) -> int | None:
    k = getattr(lm.model.generation_config, "top_k", None)
    return int(k) if k else None


@torch.no_grad()
def adapter_candidates(lm: LM, ctrl, prefix: list[int], rows: list[int], sampling, a, room: int, min_new: int, done: int,
                       gen: torch.Generator, proc=None):
    """One unit per row, row r sampled from the 3B with adapter rows[r] (-1 = base) under the 3B's own sampler.  The
    prefix minus its last token is encoded once by the base; the adapter is on from the last prefix token onward (the
    training scope).  Stops per row like SentenceEndBatch (unit rule, >= 4 new tokens, EOS); EOS is suppressed for the
    first 2 tokens (min_new_tokens=2, as in _generate).  -> (candidates, tokens computed)."""
    R, eos = len(rows), set(lm.eos_ids)
    cap = a.sent_max_tokens if a.unit == "sent" else a.chunk_max_tokens
    max_new = max(2, min(cap, room))
    ctrl.set_rows(None)
    cache = lm.model(input_ids=torch.tensor([prefix[:-1]], device=lm.device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(R)
    ctrl.set_rows(rows)
    cur = torch.full((R, 1), prefix[-1], dtype=torch.long, device=lm.device)
    out: list[list[int]] = [[] for _ in range(R)]
    alive, saw_eos = [True] * R, [False] * R
    eos_t = torch.tensor(sorted(eos), device=lm.device)
    top_k, steps = _sampler_top_k(lm), 0
    try:
        for t in range(max_new):
            logits = lm.model(input_ids=cur, past_key_values=cache, use_cache=True).logits[:, -1]
            if t < 2:
                logits[:, eos_t] = float("-inf")
            if proc is not None:                                         # the push, on the adapter's own logits
                full = torch.tensor([prefix + o + [lm.pad_id] * (t - len(o)) for o in out], dtype=torch.long, device=lm.device)
                logits = proc(full, logits.float())
            lp = processed_logprobs(logits, sampling.temperature, sampling.top_p, top_k)
            nxt = torch.multinomial(lp.exp(), 1, generator=gen)
            steps += 1
            for r in range(R):
                if not alive[r]:
                    continue
                tk = int(nxt[r, 0])
                if tk in eos:
                    alive[r], saw_eos[r] = False, True; continue
                out[r].append(tk)
                if len(out[r]) >= 4:
                    txt = lm.tokenizer.decode(out[r], skip_special_tokens=True)
                    if (semwm.first_boundary(txt) is not None) if a.unit == "sent" else (len(txt.split()) > a.W):
                        alive[r] = False
            if not any(alive):
                break
            cur = nxt
    finally:
        ctrl.set_rows(None)
    cands = []
    for r in range(R):
        cid, text, found = _trim_unit(lm.tokenizer, out[r], a)
        if not cid or not text:
            continue
        ends = saw_eos[r] and not found
        if ends and done + len(cid) < min_new:
            continue
        cands.append({"ids": cid, "text": text, "ends": ends, "found": bool(found), "adapter": int(rows[r])})
    return cands, R * steps


@torch.no_grad()
def unit_logprobs(lm: LM, ctrl, prefix: list[int], cands: list[list[int]], adapters: list[int], sampling) -> np.ndarray:
    """[n_cands, n_adapters] total processed log-probability of each candidate under each adapter (-1 = base), with
    the same scope as sampling (prefix[:-1] by the base, adapter from the last prefix token)."""
    pairs = [(i, j) for i in range(len(cands)) for j in range(len(adapters))]
    ctrl.set_rows(None)
    cache = lm.model(input_ids=torch.tensor([prefix[:-1]], device=lm.device), use_cache=True).past_key_values
    cache.batch_repeat_interleave(len(pairs))
    Lc = max(len(c) for c in cands)
    inp = torch.full((len(pairs), Lc), lm.pad_id, dtype=torch.long, device=lm.device)
    att = torch.zeros((len(pairs), len(prefix) - 1 + Lc), dtype=torch.long, device=lm.device)
    att[:, :len(prefix) - 1] = 1
    for r, (i, _) in enumerate(pairs):
        seq = [prefix[-1]] + cands[i][:-1]
        inp[r, :len(seq)] = torch.tensor(seq, device=lm.device); att[r, len(prefix) - 1:len(prefix) - 1 + len(seq)] = 1
    ctrl.set_rows([adapters[j] for _, j in pairs])
    try:
        logits = lm.model(input_ids=inp, attention_mask=att, past_key_values=cache, use_cache=True).logits
    finally:
        ctrl.set_rows(None)
    out = np.zeros((len(cands), len(adapters)))
    top_k = _sampler_top_k(lm)
    for r, (i, j) in enumerate(pairs):
        L = len(cands[i])
        lg = logits[r, :L].clone()
        lg[:2, torch.tensor(lm.eos_ids, device=lg.device)] = float("-inf")     # the sampler's min_new_tokens=2 EOS suppression
        lp = processed_logprobs(lg, sampling.temperature, sampling.top_p, top_k)
        out[i, j] = float(lp.gather(-1, torch.tensor(cands[i], device=lp.device)[:, None]).sum())
    return out


@torch.no_grad()
def gen_chain(target: LM, drafter: LM | None, ctx: list[int], key, planes, emb, sampling, seed: int, min_new: int, max_new: int, a,
              boundary=None, adapters=None):
    """SemStamp's chain with a bounded budget: every unit's mask is seeded by the previous unit's bin (as the detector
    does); per unit N candidates, keep the one deepest inside the mask (or closest to it) -- no margin, no retry loop.
    The first unit seeds the chain and is a plain sample.  All candidates' projections are logged so any smaller
    budget / other keep-rule can be replayed offline (scripts/ss_budget.py)."""
    ids: list[int] = []
    prev_bin, ended, u = None, False, 0
    st = {"units": [], "t_gen": 0, "t_score": 0, "d_gen": 0, "repairs": 0, "fallback": 0}
    steer_acc: dict[str, float] = {}
    cand_log: list[dict] = []
    gen = torch.Generator(device=target.device); gen.manual_seed(seed_from(seed, "repair"))
    while len(ids) < max_new and not ended:
        if a.eot_at_boundary and ids and len(ids) >= min_new:
            # 2026-09-29 (code): the model may end the text at any unit boundary -- one draw of its next token from the
            # plain sampler (the push leaves boundary/EOS mass unchanged); an EOS ends the text here.  Without this, an end
            # of turn right after a line (units stop AT a newline) was never sampled: the next unit call suppresses EOS for
            # 2 tokens and the model opened a fake new chat turn.
            torch.manual_seed(seed_from(seed, "end", u))
            t1 = _generate(target, ctx + ids, sampling, 1, 0)[0]
            st["t_gen"] += 1; st["end_checks"] = st.get("end_checks", 0) + 1
            if t1 and t1[0] in target.eos_ids:
                ended = True; st["ended_at_boundary"] = True
                break
        torch.manual_seed(seed_from(seed, "unit", u)); u += 1
        mask = semwm.valid_mask(key, prev_bin, a.lmbd) if prev_bin is not None else None
        prefix, room = ctx + ids, max_new - len(ids)
        if adapters is not None and mask is not None:
            # amendment 11: candidates from the bin adapters, blocks spread over the mask's bins (2+2 for lambda 0.25),
            # same stop rule (in mask AND >= stop margin) and cap; keep the deepest, or --select sir
            step, cap = (a.block or 4), a.best_of
            cands, P = [], np.zeros((0, planes.d))
            aproc = None
            if a.steer != "none" and a.beta > 0:                         # adapters + push (2026-09-26)
                aproc = semwm.NucleusSteerProcessor(planes, emb, target.tokenizer, len(prefix), mask[0], a.beta, boundary,
                                                    sampling.temperature, sampling.top_p, mode=a.steer, stat="seg", topk=a.steer_topk,
                                                    clip_sd=a.steer_clip, target_depth=a.steer_target, rep_n=a.rep_n, rep_L0=len(ctx),
                                                    seg_W=None, eos_ids=target.eos_ids + [target.pad_id], target_bins=mask)
            while len(cands) < cap:
                nb = min(step, cap - len(cands))
                rows = [int(prev_bin)] * nb if a.adapter_index == "mask" else [int(mask[i % len(mask)]) for i in range(nb)]
                new, spent = adapter_candidates(target, adapters, prefix, sorted(rows), sampling, a, room, min_new, len(ids), gen, aproc)
                st["t_gen"] += spent
                if not new:
                    break
                Pn = planes.projections(emb.encode([c["text"] for c in new]))
                cands += new; P = np.vstack([P, Pn])
                if any(semwm.signed_distance(q, mask, planes) > 0 and np.min(np.abs(q)) >= a.stop_margin for q in Pn):
                    break
            if not cands:
                break
            dep = [semwm.signed_distance(q, mask, planes) for q in P]
            region = [i for i, q in enumerate(P) if dep[i] > 0 and np.min(np.abs(q)) >= a.stop_margin]
            k = max(region, key=lambda i: dep[i]) if region else int(np.argmax(dep))
            rec = {"chosen": k, "n_drawn": len(cands), "cands": [[float(x) for x in q] + [len(c["ids"]), int(c["found"]), c["adapter"]] for q, c in zip(P, cands)]}
            if a.select == "sir" and region:
                L = unit_logprobs(target, adapters, prefix, [cands[i]["ids"] for i in region], [-1] + [int(b) for b in mask], sampling)
                st["t_score"] += (1 + len(mask)) * sum(len(cands[i]["ids"]) for i in region)
                logq = np.logaddexp.reduce(L[:, 1:], axis=1) - np.log(len(mask))
                logw = L[:, 0] - logq
                if np.isfinite(logw).any():
                    w = np.exp(logw - np.max(logw[np.isfinite(logw)])); w[~np.isfinite(logw)] = 0.0
                    j = int(torch.multinomial(torch.tensor(w / w.sum()), 1, generator=torch.Generator().manual_seed(seed_from(seed, "sir", u))))
                    k = region[j]
                    rec.update({"chosen": k, "sir_n": len(region), "sir_ess": float(w.sum() ** 2 / (w ** 2).sum()),
                                "sir_logw": [float(x) for x in logw]})
            chosen, q_ch = cands[k], P[k]
            if aproc:
                for k2, v2 in aproc.summary().items():
                    steer_acc[k2] = steer_acc.get(k2, 0.0) + v2
        elif drafter is None or mask is None:
            # --block B: draw B candidates at a time and stop at the first block holding an in-mask one (cap --best-of);
            # --block 0: one batch of --best-of.  Keep the deepest drawn candidate either way.
            cap = a.best_of if mask is not None else 1
            step = a.block if (a.block and mask is not None) else cap
            cands, P = [], np.zeros((0, planes.d))
            proc = None
            if a.steer != "none" and a.beta > 0 and mask is not None:     # E-style push toward the nearest bin of this unit's mask
                proc = semwm.NucleusSteerProcessor(planes, emb, target.tokenizer, len(prefix), mask[0], a.beta, boundary,
                                                   sampling.temperature, sampling.top_p, mode=a.steer, stat="seg", topk=a.steer_topk,
                                                   clip_sd=a.steer_clip, target_depth=a.steer_target, rep_n=a.rep_n, rep_L0=len(ctx),
                                                   seg_W=None, eos_ids=target.eos_ids + [target.pad_id], target_bins=mask)
            first_tries = 0
            while len(cands) < cap:
                # the first sentence is a single plain draw; if it is dropped (the model ended the text before min_new),
                # redraw up to 8 times, the last time without the min_new rule, instead of returning an empty text
                mn = 0 if (mask is None and first_tries == 8) else min_new
                new, spent = unit_candidates(target, prefix, [], sampling, min(step, cap - len(cands)), a, room, mn, len(ids),
                                             [proc] if proc else None)
                st["t_gen"] += spent
                if not new:
                    if mask is None and first_tries < 8:
                        first_tries += 1; st["first_redraws"] = st.get("first_redraws", 0) + 1
                        continue
                    break
                Pn = planes.projections(emb.encode([c["text"] for c in new]))
                cands += new; P = np.vstack([P, Pn])
                if mask is None or any(semwm.signed_distance(q, mask, planes) > 0 and np.min(np.abs(q)) >= a.stop_margin for q in Pn):
                    break
            if not cands:
                break
            dep = [semwm.signed_distance(q, mask, planes) if mask is not None else 0.0 for q in P]
            clear = [i for i, q in enumerate(P) if dep[i] > 0 and np.min(np.abs(q)) >= a.stop_margin]
            k = max(clear, key=lambda i: dep[i]) if (a.stop_margin > 0 and clear) else int(np.argmax(dep))
            rec = {"chosen": k, "n_drawn": len(cands), "cands": [[float(x) for x in q] + [len(c["ids"]), int(c["found"])] for q, c in zip(P, cands)]}
            if a.log_cands:                                              # bin-adapter corpus: every candidate, text and ids (unit 0 = seed)
                cand_log.append({"unit": u - 1, "prefix_len": len(prefix), "ctx_len": len(ctx), "prev_bin": prev_bin,
                                 "mask": list(mask) if mask is not None else None,
                                 "chosen": k, "cands": [{"ids": c["ids"], "text": c["text"], "proj": [float(x) for x in q],
                                                         "bin": planes.bin_of(q), "found": bool(c["found"]), "ends": bool(c["ends"])}
                                                        for q, c in zip(P, cands)]})
            if a.log_half and mask is not None:                          # early-abort replay: the first half of every candidate
                halves, htok = [], []
                for c in cands:
                    if a.unit == "seg":
                        h = " ".join(c["text"].split()[: a.W // 2])
                        n_h = next((n for n in range(1, len(c["ids"]) + 1) if len(target.decode(c["ids"][:n]).split()) >= a.W // 2), len(c["ids"]))
                    else:
                        n_h = max(1, len(c["ids"]) // 2); h = target.decode(c["ids"][:n_h]).strip() or c["text"]
                    halves.append(h); htok.append(n_h)
                Ph = planes.projections(emb.encode(halves))
                rec["half"] = [[float(x) for x in q] + [n] for q, n in zip(Ph, htok)]
            chosen, q_ch = cands[k], P[k]
            if proc:
                for k2, v2 in proc.summary().items():
                    steer_acc[k2] = steer_acc.get(k2, 0.0) + v2
        elif a.draft_v == 2:
            dsampling = SimpleNamespace(temperature=a.draft_temperature or sampling.temperature, top_p=a.draft_top_p or sampling.top_p)
            chosen, q_ch, rec = draft_verify2(target, drafter, prefix, mask, planes, emb, sampling, dsampling, a, room, min_new, len(ids), st, gen)
        else:
            chosen, q_ch, rec = draft_verify(target, drafter, prefix, mask, planes, emb, sampling, a, room, min_new, len(ids), st, gen)
        b = planes.bin_of(q_ch)
        rec.update({"prev_bin": prev_bin, "bin": b, "in_mask": bool(mask is not None and b in mask),
                    "depth": float(semwm.signed_distance(q_ch, mask, planes)) if mask is not None else None, "n_tok": len(chosen["ids"])})
        st["units"].append(rec)
        ids += chosen["ids"]; ended = chosen["ends"]; prev_bin = b
    m = [r for r in st["units"] if r["prev_bin"] is not None]
    if steer_acc:                                                        # per-unit averages of the push mechanics; counts summed
        n_p = max(1, len(m))
        st.update({k2: (v2 if k2 in ("steer_rep_guard_hits", "steer_embed_calls", "steer_strings", "steer_calls") else v2 / n_p)
                   for k2, v2 in steer_acc.items()})
    st.update({"n_tokens": len(ids), "ended": ended, "n_units": len(st["units"]),
               "in_mask_rate": float(np.mean([r["in_mask"] for r in m])) if m else None})
    if a.log_cands:
        st["_cand_log"] = cand_log                                      # popped by main() before the record is written
    return ids, st


# --------------------------------------------------------------------------- #
# steering premise check (smoke only)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def premise_check(target: LM, emb: semwm.Embedder, etok, planes: semwm.Hyperplanes, key: semwm.Key, rows, sampling, a) -> dict:
    """Same prompt, same seeds, several betas: how far does the push move a sentence's projection onto the target
    direction, and does the token table predict it?  Reported in units of the unsteered sd."""
    out = {"betas": a.premise_betas, "cases": []}
    lex = (etok.norm(dim=1) > 0).cpu().numpy()
    for p in rows[: a.premise_prompts]:
        ctx = target.chat_prefix_ids(GENERATOR_SYSTEM, p.prompt)
        prev_bin = planes.bin_of(planes.projections(emb.encode([p.prompt]))[0])
        for target_bin in semwm.valid_mask(key, prev_bin, a.lmbd):
            signs = planes.bin_signs(target_bin)
            u0 = semwm.target_direction(planes, signs, np.ones(planes.d, dtype=bool))
            table_u = (etok.cpu().numpy() @ u0)
            case = {"prompt_id": p.prompt_id, "target_bin": int(target_bin), "per_beta": {}}
            for beta in a.premise_betas:
                p_u, pred, acc, capped, matched = [], [], [], [], []
                for c in range(a.premise_calls):
                    proc = semwm.SteerBiasProcessor(etok, planes, emb, target.tokenizer, len(ctx), target_bin, beta, a.k, a.margin,
                                                    eos_ids=target.eos_ids, zero_ids=[target.pad_id]) if beta > 0 else None
                    torch.manual_seed(seed_from("premise", p.prompt_id, target_bin, c))
                    rows_ = _generate(target, ctx, sampling, a.sent_max_tokens, 2, [proc] if proc else None,
                                      semwm.SentenceEndBatch(target.tokenizer, len(ctx), target.eos_ids), a.n_cand)
                    for row in rows_:
                        raw, _ = clean_ids(target, row)
                        cand, piece, found = semwm.trim_candidate(target.tokenizer, raw, [])
                        if not cand:
                            continue
                        e = emb.encode([piece])[0]
                        pr = planes.projections(e[None])[0]
                        p_u.append(float(e @ u0)); capped.append(not found)
                        toks = [t for t in cand if lex[t]]
                        pred.append(float(np.mean(table_u[toks])) if toks else 0.0)
                        acc.append(bool(planes.bin_of(pr) == target_bin and planes.margin_ok(pr, a.margin)))
                        matched.append(int(np.sum(signs * pr >= a.margin)))
                case["per_beta"][str(beta)] = {"n": len(p_u), "p_u_mean": float(np.mean(p_u)), "p_u_sd": float(np.std(p_u)),
                                              "accept_rate": float(np.mean(acc)), "planes_matched": float(np.mean(matched)),
                                              "capped": float(np.mean(capped)),
                                              "r_pred": float(np.corrcoef(pred, p_u)[0, 1]) if len(p_u) > 3 and np.std(pred) > 0 else None}
            null_sd = case["per_beta"][str(float(a.premise_betas[0]))]["p_u_sd"] or 0.036
            for beta, r in case["per_beta"].items():
                r["delta_null_sd"] = (r["p_u_mean"] - case["per_beta"][str(float(a.premise_betas[0]))]["p_u_mean"]) / null_sd
            out["cases"].append(case)
            print(f"[premise] {p.prompt_id[:28]} bin {target_bin}: " + " | ".join(
                f"b={b}: d={r['delta_null_sd']:+.2f}sd acc={r['accept_rate']:.2f} matched={r['planes_matched']:.2f} r={r['r_pred'] if r['r_pred'] is None else round(r['r_pred'], 2)}"
                for b, r in case["per_beta"].items()), flush=True)
    agg = {}
    for beta in a.premise_betas:
        rs = [c["per_beta"][str(beta)] for c in out["cases"]]
        agg[str(beta)] = {k: float(np.mean([r[k] for r in rs if r[k] is not None])) for k in ("delta_null_sd", "accept_rate", "planes_matched", "capped")}
        agg[str(beta)]["r_pred"] = float(np.mean([r["r_pred"] for r in rs if r["r_pred"] is not None])) if any(r["r_pred"] is not None for r in rs) else None
    out["aggregate"] = agg
    stop = all(agg[str(b)]["delta_null_sd"] < 0.5 for b in a.premise_betas if b > 0) and \
        (agg[str(float(a.premise_betas[0]))]["r_pred"] is None or agg[str(float(a.premise_betas[0]))]["r_pred"] < 0.1) and max(agg[str(b)]["accept_rate"] for b in a.premise_betas) < 0.15
    out["stop_rule_triggered"] = bool(stop)
    print(f"[premise] aggregate: {json.dumps(agg)}\n[premise] STOP RULE {'TRIGGERED' if stop else 'not triggered'}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "semstamp", "steered_sent", "steered_seg", "chain"], required=True)
    ap.add_argument("--draft", choices=["none", "assist", "loop", "nucleus"], default="none")
    ap.add_argument("--unit", choices=["sent", "seg"], default="sent", help="chain arm: sentence or W-word segment units")
    ap.add_argument("--verify-top", type=int, default=4, help="chain + nucleus drafts: drafts checked by the 3B per round")
    ap.add_argument("--max-repairs", type=int, default=3)
    ap.add_argument("--block", type=int, default=0, help="chain: draw candidates in blocks of this size, stop at the first block with an in-mask one")
    ap.add_argument("--proposal", choices=["base", "adapters"], default="base", help="chain: where candidates come from (amendment 11)")
    ap.add_argument("--adapters", default="runs/ss_adapters/v1/adapters.pt")
    ap.add_argument("--select", choices=["deepest", "sir"], default="deepest", help="adapters: keep the deepest in-region candidate, or resample by p/q_mix")
    ap.add_argument("--stop-margin", type=float, default=0.0, help="chain: a block stops the unit only on an in-mask candidate at least this far from every hyperplane")
    ap.add_argument("--log-half", action="store_true", help="chain: also log the projection of each candidate's first half (early-abort replay)")
    ap.add_argument("--draft-v", type=int, default=1, help="1B-draft path: 1 = smoke version, 2 = with the three fixes")
    ap.add_argument("--draft-temperature", type=float, default=0.0, help="v2: the 1B's own temperature (0 = same as the 3B)")
    ap.add_argument("--draft-top-p", type=float, default=0.0)
    ap.add_argument("--arm-label", default="")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--models-json", default="assets/ss_models.json")
    ap.add_argument("--run-name", default="ss_llama3b")
    ap.add_argument("--target", default="llama3b", help="generator key in assets/ss_models.json (llama3b, llama1b, qwen08b)")
    ap.add_argument("--source-run", default="runs/conf_cross")
    ap.add_argument("--prompts", choices=["held_out", "null_calibration", "ss_train_train", "ss_train_val", "ob_bench", "sp_bench", "it_bench"], default="held_out")
    ap.add_argument("--ob-domains", nargs="*", default=None, help="ob_bench: restrict to these domains (e.g. ob_arxiv ob_booksum ob_wikipedia)")
    ap.add_argument("--adapter-index", choices=["bin", "mask"], default="bin",
                    help="adapters: bin = one adapter per target bin (amendment 11); mask = one per mask, indexed by the previous unit's bin (RL/RFT)")
    ap.add_argument("--log-cands", default="", help="chain: write every candidate (ids, text, projections, bin) to <path>.shard<k>.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-domain", type=int, default=200)
    ap.add_argument("--per-domain", type=int, default=0, help="smoke: first N prompts of each domain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=4)
    ap.add_argument("--id-n-cand", type=int, default=None, help="semstamp: hash text ids with this n_cand (batch size only changes "
                    "speed when max_trials is a multiple of it: the first passing sentence in draw order is kept either way)")
    ap.add_argument("--max-trials", type=int, default=100)
    ap.add_argument("--margin", type=float, default=semwm.MARGIN)
    ap.add_argument("--lmbd", type=float, default=semwm.LMBD)
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--sent-max-tokens", type=int, default=64)
    ap.add_argument("--best-of", type=int, default=0, help="steered_seg with --beta 0: N exact samples per W-word chunk, keep the deepest")
    ap.add_argument("--chunk-max-tokens", type=int, default=48)
    ap.add_argument("--steer", choices=["none", "table", "marginal"], default="none", help="round-2 nucleus steerer (NucleusSteerProcessor)")
    ap.add_argument("--steer-stat", choices=["seg", "dir"], default="seg", help="per-unit score pushed / selected on")
    ap.add_argument("--steer-topk", type=int, default=20)
    ap.add_argument("--steer-clip", type=float, default=1.0, help="clamp of the standardised gain, in sd")
    ap.add_argument("--steer-target", type=float, default=0.05, help="depth at which a unit stops being pushed (detector clip)")
    ap.add_argument("--rep-n", type=int, default=4)
    ap.add_argument("--embedder", default=SEMSTAMP_SBERT)
    ap.add_argument("--embedder-prefix", default="")
    ap.add_argument("--etok-cache", default="assets/etok_llama32_semstamp_sbert.npz")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--plain", choices=["", "semstamp", "excerpt"], default="", help="base models, no chat template: continue "
                    "SemStamp's prompt (first sentence of the document, sampling_utils.extract_prompt_from_text) or the benchmark's excerpt")
    ap.add_argument("--temperature", type=float, default=0.0, help="override the config's temperature")
    ap.add_argument("--top-p", type=float, default=0.0, help="override the config's top-p (1.0 = off)")
    ap.add_argument("--top-k", type=int, default=None, help="top-k for the sampler (0 = off; default: the model's generation config)")
    ap.add_argument("--rep-penalty", type=float, default=0.0, help="HF repetition_penalty (SemStamp: 1.05)")
    ap.add_argument("--join-leading-newline", action="store_true", help="sentence units: a line break at the start of a unit "
                    "joins the next sentence instead of being a unit (OPT; runs before 2026-09-30 ran without it)")
    ap.add_argument("--min-new", type=int, default=-1, help="minimum text length in tokens (default: the config's min_new_tokens, 64)")
    ap.add_argument("--eot-at-boundary", action="store_true", help="chain and semstamp arms: the model may end the text at every unit boundary "
                                                                     "(needed for code; prose runs before 2026-09-29 ran without it)")
    ap.add_argument("--premise-check", action="store_true")
    ap.add_argument("--premise-prompts", type=int, default=2)
    ap.add_argument("--premise-calls", type=int, default=8)
    ap.add_argument("--premise-betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    a = ap.parse_args()
    if a.arm == "steered_seg" and a.best_of > 0 and a.beta > 0 and a.steer == "none":
        ap.error("--best-of with --beta > 0 needs --steer table|marginal (the round-1 processor is not used for selection)")
    if a.draft == "assist" and a.n_cand > 1:
        a.n_cand = 1                                            # assisted generation cannot batch candidates
    label = a.arm_label or (f"chain_{a.unit}_" + (f"blk{a.block}x{a.best_of}" if a.block else f"bo{a.best_of}") + (f"_push{a.beta:g}" if a.steer != "none" and a.beta > 0 else "") + ("_adapt" if a.proposal == "adapters" else "") + ("_sir" if a.select == "sir" else "") + (f"_m{a.stop_margin:g}" if a.stop_margin > 0 else "") + (("_d1b" + (f"2_t{a.draft_temperature:g}_p{a.draft_top_p:g}" if a.draft_v == 2 else "")) if a.draft == "nucleus" else "") if a.arm == "chain" else f"bestof{a.best_of}_seg" if a.best_of else a.arm if a.draft == "none" else f"{a.arm}_draft" if a.draft == "loop" else f"{a.arm}_assist")

    semwm.LEAD_JOIN = bool(a.join_leading_newline)
    cfg = load_config(a.config)
    m = json.loads(Path(a.models_json).read_text())
    gcfg = cfg.generation_config(None, {})
    sampling = gcfg.sampling()
    if a.temperature or a.top_p or a.top_k is not None or a.rep_penalty:   # e.g. SemStamp's decoding: T 0.7, top-k 0, rep 1.05
        sampling = SimpleNamespace(temperature=a.temperature or sampling.temperature, top_p=a.top_p or sampling.top_p,
                                   top_k=sampling.top_k, hf_top_k=a.top_k, repetition_penalty=a.rep_penalty)
    min_new, max_new = (a.min_new if a.min_new >= 0 else gcfg.min_new_tokens), (a.max_new or gcfg.max_new_tokens)
    rows, key_of, partition = prompts_for(cfg, a.prompts, a.limit_per_domain, a.source_run, a.ob_domains)
    if a.per_domain:
        by_dom = defaultdict(list)
        for r in rows:
            by_dom[r.domain].append(r)
        rows = sorted([r for d in sorted(by_dom) for r in by_dom[d][: a.per_domain]], key=lambda r: r.prompt_id)
    if a.limit:
        rows = rows[: a.limit]
    mine = [p for i, p in enumerate(rows) if i % a.num_shards == a.shard]
    t0 = time.time()
    target = LM(m[a.target]["path"], a.dtype)
    drafter = LM(m["llama1b"]["path"], a.dtype) if a.draft != "none" else None
    adapter_ctrl = None
    if a.arm == "chain" and a.proposal == "adapters":
        from wmworld import multilora
        adapter_ctrl, ameta = multilora.load(a.adapters, target.model)
        assert ameta["key"] == semwm.Key.shipped().label(), f"adapters trained for key {ameta['key']}"
        print(f"loaded bin adapters {a.adapters} (key {ameta['key']}, corpus {ameta.get('corpus_hash')})", flush=True)
    drafter_ratio = (sum(p.numel() for p in drafter.model.parameters()) / sum(p.numel() for p in target.model.parameters())) if drafter else 0.0
    key = semwm.Key.shipped()
    emb = etok = planes = boundary = None
    if a.arm != "base":
        emb = semwm.Embedder(a.embedder, device=str(target.device), batch_size=64, prefix=a.embedder_prefix)
        planes = semwm.Hyperplanes.for_key(key, emb.dim)
        if a.arm.startswith("steered") and a.beta > 0:
            etok = semwm.token_table(emb, target.tokenizer, a.etok_cache, log=lambda s: print(s, flush=True))
        if a.steer != "none" and a.beta > 0:
            boundary = semwm.boundary_token_mask(target.tokenizer, target.eos_ids + [target.pad_id], target.device,
                                                 target.model.get_output_embeddings().weight.shape[0])
            print(f"boundary class: {int(boundary.sum())} of {len(boundary)} tokens", flush=True)
    print(f"loaded target {m[a.target]['path']}" + (f" + drafter {m['llama1b']['path']}" if drafter else "") + f" ({time.time() - t0:.0f}s)", flush=True)

    params = {"arm": a.arm, "draft": a.draft, "beta": a.beta if (a.arm.startswith("steered") or a.steer != "none") else 0.0, "k": a.k, "n_cand": a.n_cand,
              "max_trials": a.max_trials, "margin": a.margin, "lmbd": a.lmbd, "W": a.W, "sent_max_tokens": a.sent_max_tokens, "best_of": a.best_of,
              "unit": a.unit, "block": a.block, "stop_margin": a.stop_margin, "proposal": a.proposal,
              "adapters": a.adapters if a.proposal == "adapters" else None, "adapter_index": a.adapter_index, "select": a.select, "log_half": a.log_half, "verify_top": a.verify_top, "max_repairs": a.max_repairs, "draft_v": a.draft_v,
              "draft_temperature": a.draft_temperature, "draft_top_p": a.draft_top_p,
              "steer": a.steer, "steer_stat": a.steer_stat, "steer_topk": a.steer_topk, "steer_clip": a.steer_clip,
              "steer_target": a.steer_target, "rep_n": a.rep_n,
              "embedder": a.embedder, "key": {"rand_seed": key.rand_seed, "hash_key": key.hash_key, "lsh_dim": key.lsh_dim},
              "target": m[a.target], "drafter": m["llama1b"] if drafter else None, "dtype": a.dtype, "max_new": max_new}
    run_dir = Path("runs") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest.json", {"partition": partition, "source_run": a.source_run, "key_set": "confirmation",
                                           "generator": m[a.target]["path"], "sampling": {"temperature": sampling.temperature, "top_p": sampling.top_p,
                                                                                            "min_new_tokens": min_new, "max_new_tokens": max_new},
                                           "arms": {label: params}})
    if a.premise_check:
        out = premise_check(target, emb, etok, planes, key, rows, sampling, a)
        write_json(run_dir / "premise_check.json", out)
        return

    if a.min_new >= 0:
        params["min_new"] = min_new
    if a.eot_at_boundary:
        params["eot_at_boundary"] = True
    if a.plain:
        params["plain"] = a.plain
    if a.join_leading_newline:
        params["join_leading_newline"] = True
    if isinstance(sampling, SimpleNamespace):
        params["sampling"] = {"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.hf_top_k,
                              "repetition_penalty": sampling.repetition_penalty}
    # 2026-09-29 the SemStamp encoder and the 1B moved from /nfs-stor to /l (identical files); text ids keep the old
    # location so runs written before the move still resume
    id_params = dict(params, n_cand=a.id_n_cand) if a.id_n_cand is not None else params
    if a.id_n_cand is not None:
        assert a.max_trials % a.n_cand == 0 and a.max_trials % a.id_n_cand == 0, "--id-n-cand needs max_trials divisible by both"
    ch = content_hash({"params": json.loads(json.dumps(id_params).replace("/l/users/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert",
                                                                       "/nfs-stor/daniil.orel/hf_cache/hub/models--AbeHou--SemStamp-c4-sbert")
                                                  .replace("/l/users/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct",
                                                           "/nfs-stor/daniil.orel/hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct")),
                       "seed": a.seed}, length=12)
    path = run_dir / f"generations.{label}.shard{a.shard}.jsonl"
    repair_jsonl(path)
    w = JsonlWriter(path)
    # texts already written by ANY shard file of this arm (so an arm can be resumed with a different --num-shards)
    done_elsewhere = set()
    for f in run_dir.glob(f"generations.{label}.shard*.jsonl"):
        if f != path:
            for ln in open(f):
                try:
                    done_elsewhere.add(json.loads(ln)["id"])
                except Exception:
                    pass
    cand_path = Path(f"{a.log_cands}.shard{a.shard}.jsonl") if a.log_cands else None
    if cand_path:
        cand_path.parent.mkdir(parents=True, exist_ok=True)
    for n, p in enumerate(mine, 1):
        oid = output_id(a.run_name, label, p.group_id, "shipped" if a.arm != "base" else None, a.seed, ch)
        if oid in w or oid in done_elsewhere:
            continue
        ptext = {"semstamp": lambda: semstamp_prompt(p.excerpt + p.human_continuation), "excerpt": lambda: p.excerpt,
                 "": lambda: p.prompt}[a.plain]()
        ctx = target.plain_prefix_ids(ptext) if a.plain else target.chat_prefix_ids(GENERATOR_SYSTEM, p.prompt)
        seed = seed_from(a.run_name, label, p.prompt_id, a.seed)
        target.n_forward = 0
        if drafter:
            drafter.n_forward = 0
        emb_calls0 = emb.n_calls if emb else 0
        t1 = time.time()
        try:
            if a.arm == "base":
                ids, stats = gen_whole(target, ctx, sampling, seed, min_new, max_new, None, drafter if a.draft == "assist" else None)
            elif a.arm == "chain":
                ids, stats = gen_chain(target, drafter if a.draft == "nucleus" else None, ctx, key, planes, emb, sampling, seed, min_new, max_new, a,
                                       boundary, adapter_ctrl)
                stats["param_ratio"] = drafter_ratio
            elif a.arm == "steered_seg" and a.best_of > 0:
                mk = (lambda L0, c=ctx: nucleus_proc(a, target, emb, etok, planes, key, boundary, sampling, L0, len(c), None)) \
                    if (a.steer != "none" and a.beta > 0) else None
                ids, stats = gen_bestof_chunks(target, ctx, key, planes, emb, sampling, seed, min_new, max_new, a, mk)
            elif a.arm == "steered_seg" and a.steer != "none":
                proc = nucleus_proc(a, target, emb, etok, planes, key, boundary, sampling, len(ctx), len(ctx), a.W) if a.beta > 0 else None
                ids, stats = gen_whole(target, ctx, sampling, seed, min_new, max_new, proc)
                if proc:
                    stats.update(proc.summary())
            elif a.arm == "steered_seg":
                proc = semwm.SteerBiasProcessor(etok, planes, emb, target.tokenizer, len(ctx), semwm.fixed_bin(key, a.lmbd), a.beta, a.k,
                                                a.margin, eos_ids=target.eos_ids, zero_ids=[target.pad_id], seg_W=a.W) if a.beta > 0 else None
                ids, stats = gen_whole(target, ctx, sampling, seed, min_new, max_new, proc, drafter if a.draft == "assist" else None)
                if proc:
                    stats.update({k: v for k, v in proc.stats.items()})
            else:
                ids, stats = gen_sentencewise(target, drafter, ctx, ptext, key, planes, emb, etok, sampling, seed, min_new, max_new, a)
            text, status, err = normalize_text(target.decode(ids)), "ok", None
        except Exception as e:
            ids, stats, text, status, err = [], {}, "", "failed", repr(e)
        stats.update({"wall_seconds": time.time() - t1, "target_forward_calls": target.n_forward,
                      "drafter_forward_calls": drafter.n_forward if drafter else 0, "n_embed_calls": (emb.n_calls - emb_calls0) if emb else 0})
        cand_log = stats.pop("_cand_log", None)
        if cand_path is not None and cand_log is not None and status == "ok":
            # one line per text, written before its generation record: readers keep the LAST line of each prompt_id
            with cand_path.open("a") as fh:
                fh.write(json.dumps({"id": oid, "prompt_id": p.prompt_id, "domain": p.domain, "seed": seed, "units": cand_log}) + "\n")
        w.write(GenerationRecord(
            id=oid, stage=a.run_name, arm=label, group_id=p.group_id, prompt_id=p.prompt_id, domain=p.domain,
            key_id=("shipped" if a.arm != "base" else None), seed=a.seed, text=text, text_hash=text_hash(text),
            accepted_tokens=len(ids), visible_tokens=len(ids), stop_reason=("eos" if stats.get("ended") or len(ids) < max_new else "max_new_tokens"),
            ends_mid_sentence=ends_mid_sentence(text), config_hash=ch, status=status, error=err, cost=stats,
            assigned_key_id=key_of.get(p.prompt_id), arm_config=params))
        if n % 5 == 0 or n == len(mine):
            brief = {k: stats.get(k) for k in ("n_tokens", "trials", "accepted", "maxed_out", "left_region_after_verify", "chunks", "depth_chosen", "depth_mean", "align_ok", "align_units", "steer_bias_abs_mean", "steer_p_boundary_before", "steer_p_boundary_after", "steer_rep_guard_hits", "n_units", "in_mask_rate", "t_gen", "t_score", "d_gen", "repairs", "fallback", "wall_seconds") if k in stats}
            print(f"[{label} shard {a.shard}] {n}/{len(mine)} ({time.time() - t0:.0f}s) {brief}", flush=True)
    w.close()
    print(f"[{label} shard {a.shard}] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
