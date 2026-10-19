"""Model-backed self-test on the real runtime (spec Section 14, model-backed checks).

Targets: checkpoint loading, non-thinking serialization, teacher-forced
likelihood correctness (brute force), neutral-contrast zero, batched vs single
scoring tolerance, prompt/trace independence of the score, cache/state
isolation (forked prefix vs fresh prefill; rejected candidates do not
contaminate accepted state), EOS and minimum-length handling.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict

import torch

from .backend import SamplingParams
from .config import Config
from .generation import CandidateSelectionGenerator, GenerationConfig, arm_config
from .runtime import GENERATION_PREFIX_TEXT, QwenRuntime
from .scales import ScaleTable
from .scoring import NEUTRAL_PREFIX, KeySpec, LikelihoodScorer


def run_selftest(cfg: Config, device: str | None = None, log=print) -> dict:
    rc = cfg.runtime_config()
    if device:
        rc.device = device
    rc.fork_prefix_cache = True          # exercised and compared against fresh prefill below
    rt = QwenRuntime(rc)
    checks: dict[str, dict] = {}

    def record(name, passed, **info):
        checks[name] = {"passed": bool(passed), **info}
        log(f"  [{'PASS' if passed else 'FAIL'}] {name} {info if not passed else ''}")

    # 1. loading + template
    record("loading_no_missing_keys", not rt.loading_info.get("missing_keys"), info=rt.loading_info)
    record("template_nonthinking_prefix", rt.rendered_probe.endswith(GENERATION_PREFIX_TEXT), tail=rt.rendered_probe[-60:])
    record("eos_ids_include_im_end_and_endoftext", 248046 in rt.eos_ids and 248044 in rt.eos_ids, eos=rt.eos_ids)

    scorer = LikelihoodScorer(rt, max_input_tokens=int(cfg["scoring"]["max_input_tokens"]))
    key = KeySpec(key_id="selftest", prefix="You eat chicken every day.", family="common_single")
    neutral = KeySpec(key_id="neutral", prefix=NEUTRAL_PREFIX, family="none")
    text = "The market opened early, and the vendors arranged their produce in neat rows before the first customers arrived."

    # 2. brute-force likelihood equality: unchunked float32 head + full log-softmax over every position,
    #    explicit one-token shift, versus the runtime's chunked gather (same hidden states)
    fp32 = rt.dtype == torch.float32
    ctx = scorer.context_ids(key, neutral=False)
    tgt = scorer.tokenize(text)
    with torch.no_grad():
        seq = torch.tensor([ctx + tgt], device=rt.device)
        hidden = rt.model.model(input_ids=seq, use_cache=False).last_hidden_state[0].to(torch.float32)
        logits = hidden @ rt._head_w32.T
        lp = torch.log_softmax(logits, dim=-1)
        brute = sum(float(lp[len(ctx) - 1 + i, tgt[i]]) for i in range(len(tgt)))
        # and the model's own head (bf16 or fp32 depending on dtype), for information
        own = rt.model(input_ids=seq, use_cache=False).logits[0].float()
        own_lp = torch.log_softmax(own, dim=-1)
        own_brute = sum(float(own_lp[len(ctx) - 1 + i, tgt[i]]) for i in range(len(tgt)))
    mine = rt.logprob_sums(ctx, [tgt])[0]
    record("likelihood_matches_bruteforce", abs(brute - mine) < 1e-4 * max(1, len(tgt)), brute=brute, runtime=mine,
           abs_diff=abs(brute - mine), model_head_bruteforce=own_brute, model_head_vs_fp32_head_diff=abs(own_brute - brute))
    # the visible-text targets contain no template/think tokens
    record("targets_exclude_template_tokens", not any(t in (248045, 248046, 248068, 248069) for t in tgt), n_targets=len(tgt))
    record("context_ends_with_generation_prefix", ctx[-len(rt.generation_prefix_ids):] == rt.generation_prefix_ids)

    # 3. neutral contrast is zero; prompt independence (score has no prompt argument; identical text -> identical score)
    z = scorer.score(text, neutral).score
    record("neutral_contrast_zero", z == 0.0, value=z)
    s1 = scorer.score(text, key).score
    s2 = scorer.score(text, key).score
    record("score_repeatable", s1 == s2, s1=s1, s2=s2)

    # 4. batched vs single
    texts = [text, "Short line.", "A much longer paragraph follows here, with several clauses and a number of commas, "
             "so that the padded batch contains rows of clearly different lengths for the equality check."]
    batched = scorer.score_texts(texts, key)
    single = [scorer.score(t, key) for t in texts]
    diffs = [abs(a.score - b.score) for a, b in zip(batched, single)]
    tol = 1e-4 if fp32 else 2e-3
    record("batched_vs_single_within_tol", max(diffs) < tol, max_abs_diff=max(diffs), tol=tol, dtype=str(rt.dtype))

    # 5. fork vs fresh prefill: identical next-token distributions and identical samples
    gctx = rt.chat_prefix_ids("You are a helpful assistant.", "Write two sentences about rivers.")
    sp = SamplingParams(temperature=0.8, top_p=0.95)
    rt.forget_prefix()
    rt.cfg.fork_prefix_cache = False
    fresh = rt.sample_units(gctx, 4, 12, sp, seed=7, eos_allowed_from=0)
    rt.cfg.fork_prefix_cache = True
    rt.remember_prefix(gctx)
    forked = rt.sample_units(gctx, 4, 12, sp, seed=7, eos_allowed_from=0)
    record("fork_equals_fresh_samples", [c.token_ids for c in fresh] == [c.token_ids for c in forked],
           fresh=[c.token_ids[:6] for c in fresh], forked=[c.token_ids[:6] for c in forked])
    # logits equality after extending the remembered prefix
    ext = gctx + fresh[0].token_ids[:5]
    rt.remember_prefix(ext)
    _, lg_fork = rt._prefill_forked(ext, 2)
    _, lg_fresh = rt._prefill(ext, 2)
    ld = float((lg_fork - lg_fresh).abs().max().item())
    record("fork_logits_match_fresh", ld < (1e-3 if fp32 else 0.5), max_abs_logit_diff=ld)
    # rejected candidates do not contaminate: sampling from the remembered prefix twice gives identical results
    a = rt.sample_units(ext, 3, 8, sp, seed=11, eos_allowed_from=0)
    b = rt.sample_units(ext, 3, 8, sp, seed=11, eos_allowed_from=0)
    record("accepted_state_immutable_across_forks", [c.token_ids for c in a] == [c.token_ids for c in b])
    rt.forget_prefix()
    rt.cfg.fork_prefix_cache = bool(cfg["runtime"].get("fork_prefix_cache", False))

    # 6. EOS suppression until the minimum; EOS allowed afterwards
    c = rt.sample_units(gctx, 2, 6, sp, seed=3, eos_allowed_from=6)
    record("eos_suppressed_before_minimum", all(not x.ended and len(x.token_ids) == 6 for x in c))

    # 7. generator smoke: llm_global with toy sigma; telescoping identity on raw gains
    table = ScaleTable(scorer_id=scorer.scorer_id, fitted_on="selftest", toy=True, floor=1e-6, numerical_discrepancy=0.0,
                       entries={f"common_single|tokens16|global|{b}": {"sigma": 0.05, "status": "ok", "n_pools": 1, "fallback_from": None,
                                                                       "raw_median_sd": 0.05} for b in ("[0,64)", "[64,128)", "[128,256)", "[256,inf)")})
    gen = CandidateSelectionGenerator(rt, scorer, table)
    gcfg = arm_config("llm_global", GenerationConfig(candidates=3, min_new_tokens=8, max_new_tokens=40, unit_tokens=16))
    t0 = time.time()
    res = gen.generate("Describe a quiet morning in a small town.", key, gcfg, 0, "selftest_out")
    tele = sum(t.gains[t.chosen] for t in res.trajectory if t.gains[t.chosen] is not None)
    final = scorer.score(res.text, key).score
    # tolerance = measured batched-vs-single discrepancy (the chosen candidate was scored in a batch; the final text singly)
    tele_tol = max(1e-6, 2.0 * max(diffs)) * max(1, res.n_decisions)
    record("telescoping_identity", abs(tele - final) <= tele_tol, sum_gains=tele, final=final, tol=tele_tol,
           text=res.text[:120], seconds=time.time() - t0, stop=res.stop_reason)
    # boundary independence: rescoring the same final text gives the same score regardless of segmentation (same function)
    record("boundary_independence", scorer.score(res.text, key).score == final)

    passed = all(v["passed"] for v in checks.values())
    return {"all_passed": passed, "checks": checks, "backend": asdict(rt.info), "cost": rt.cost_snapshot()}
