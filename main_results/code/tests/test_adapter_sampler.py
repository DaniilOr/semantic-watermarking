"""CPU tests of the bin-adapter sampling math in scripts/steered_semstamp.py (tiny random Llama, no downloads)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from wmworld import multilora

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("steered_semstamp", ROOT / "scripts" / "steered_semstamp.py")
ss = importlib.util.module_from_spec(spec)
sys.modules["steered_semstamp"] = ss
spec.loader.exec_module(ss)


@pytest.mark.parametrize("top_k", [None, 5])
def test_processed_logprobs_match_hf_warpers(top_k):
    torch.manual_seed(0)
    logits = torch.randn(6, 40) * 3
    warpers = [TemperatureLogitsWarper(0.8)] + ([TopKLogitsWarper(top_k)] if top_k else []) + [TopPLogitsWarper(0.95)]
    s = logits.clone()
    for w in warpers:
        s = w(None, s)
    want = torch.log_softmax(s, dim=-1)
    got = ss.processed_logprobs(logits, 0.8, 0.95, top_k)
    assert torch.equal(torch.isinf(got), torch.isinf(want))
    fin = torch.isfinite(want)
    assert torch.allclose(got[fin], want[fin], atol=1e-5)


def _tiny_lm():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.top_k = None
    return SimpleNamespace(model=model, device=torch.device("cpu"), pad_id=63, eos_ids=[1, 2])


def test_unit_logprobs_equal_a_full_forward_and_fresh_adapters_equal_the_base():
    lm = _tiny_lm()
    ctrl = multilora.attach(lm.model, n=3, r=2, alpha=4)
    prefix = [5, 9, 11, 7, 20]
    cands = [[30, 31, 32], [40, 41], [12, 13, 14, 15]]
    sampling = SimpleNamespace(temperature=0.8, top_p=0.95)
    L = ss.unit_logprobs(lm, ctrl, prefix, cands, [-1, 0, 2], sampling)
    assert L.shape == (3, 3)
    assert abs(L[:, 0] - L[:, 1]).max() < 1e-5 and abs(L[:, 0] - L[:, 2]).max() < 1e-5      # B = 0: q == p -> SIR weights 1
    for i, c in enumerate(cands):                                                         # cache path == plain full forward
        with torch.no_grad():
            lg = lm.model(input_ids=torch.tensor([prefix + c])).logits[0, len(prefix) - 1:len(prefix) - 1 + len(c)]
        lg[:2, torch.tensor(lm.eos_ids)] = float("-inf")
        lp = ss.processed_logprobs(lg, 0.8, 0.95, None)
        want = float(lp.gather(-1, torch.tensor(c)[:, None]).sum())
        assert L[i, 0] == pytest.approx(want, abs=1e-4)


def test_unit_logprobs_use_each_rows_own_adapter():
    lm = _tiny_lm()
    ctrl = multilora.attach(lm.model, n=3, r=2, alpha=4)
    for mod in lm.model.modules():
        if isinstance(mod, multilora.MultiLoRALinear):
            torch.nn.init.normal_(mod.B, std=0.5)
    sampling = SimpleNamespace(temperature=0.8, top_p=1.0)
    prefix, cands = [5, 9, 11, 7], [[30, 31, 32], [40, 41]]
    L = ss.unit_logprobs(lm, ctrl, prefix, cands, [-1, 0, 1], sampling)
    for j, adapter in enumerate([-1, 0, 1]):
        single = ss.unit_logprobs(lm, ctrl, prefix, cands, [adapter], sampling)
        assert abs(L[:, j] - single[:, 0]).max() < 1e-4
    assert abs(L[:, 1] - L[:, 0]).max() > 1e-3                                              # adapters change the model


# ---- RL steering head (amendment 14) ------------------------------------------------------- #
spec2 = importlib.util.spec_from_file_location("ss_rl_steer", ROOT / "scripts" / "ss_rl_steer.py")
st = importlib.util.module_from_spec(spec2); sys.modules["ss_rl_steer"] = st; spec2.loader.exec_module(st)


def _steer_setup(zero=True):
    torch.manual_seed(0)
    V, d = 100, 16
    head = st.SteerHead(d, cap=2.0)
    if not zero:
        torch.nn.init.normal_(head.net[-1].weight, std=0.5)
    emb_w = torch.randn(V, d)
    boundary = torch.zeros(V, dtype=torch.bool); boundary[[1, 2, 3]] = True
    logits = torch.randn(3, V) * 2; logits[:, 1] += 3
    return head, emb_w, boundary, logits, torch.randn(3, d), torch.tensor([0, 3, 7])


def test_steering_head_starts_as_the_base_sampler():
    head, emb_w, boundary, logits, h, mid = _steer_setup(zero=True)
    lp, lp_base = st.steered_logprobs(logits, h, mid, head, emb_w, boundary, 0.8, 0.95)
    fin = torch.isfinite(lp_base)
    assert torch.equal(torch.isfinite(lp), fin) and torch.allclose(lp[fin], lp_base[fin], atol=1e-5)


def test_steering_head_keeps_boundary_mass_nucleus_and_cap():
    head, emb_w, boundary, logits, h, mid = _steer_setup(zero=False)
    lp, lp_base = st.steered_logprobs(logits, h, mid, head, emb_w, boundary, 0.8, 0.95)
    assert torch.equal(torch.isfinite(lp), torch.isfinite(lp_base))                  # nothing outside the nucleus
    pb = lambda x: torch.where(boundary[None, :] & torch.isfinite(x), x.exp(), torch.zeros_like(x)).sum(-1)
    assert torch.allclose(pb(lp), pb(lp_base), atol=1e-5)                             # P(boundary) unchanged
    assert not torch.allclose(lp[torch.isfinite(lp)], lp_base[torch.isfinite(lp_base)])  # but the head does steer
    b = head(h, emb_w[torch.arange(10)][None].expand(3, 10, -1), mid)
    assert float(b.abs().max()) <= 2.0 + 1e-6


def test_steering_gradients_stay_finite_with_blocked_tokens_and_no_lexical_rows():
    head, emb_w, boundary, logits, h, mid = _steer_setup(zero=False)
    logits[2] = -30.0; logits[2, 1] = 10.0                                   # row 2: nucleus = one boundary token only
    block = torch.zeros_like(logits, dtype=torch.bool); block[0, [1, 2]] = True   # row 0: EOS-like tokens blocked
    lp, lp_base = st.steered_logprobs(logits, h, mid, head, emb_w, boundary, 0.8, 0.95, block)
    fin = torch.isfinite(lp_base)
    zero = torch.zeros_like(lp)
    lp_s, lb_s = torch.where(fin, lp, zero), torch.where(fin, lp_base, zero)
    kl = (lp_s.exp() * fin * (lp_s - lb_s)).sum(-1)
    (kl.mean() - lp[0, torch.isfinite(lp[0])].sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert torch.isfinite(kl).all() and float(kl[2]) == pytest.approx(0.0, abs=1e-6)
