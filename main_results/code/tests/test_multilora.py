"""Model-free tests for the multi-adapter LoRA module (bin adapters)."""

import torch
from torch import nn

from wmworld import multilora


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(8, 4)
        self.other = nn.Linear(4, 4)

    def forward(self, x):
        return self.other(self.down_proj(torch.relu(self.q_proj(x))))


def _pair(n=3):
    m0, m1 = Tiny(), Tiny()
    m1.load_state_dict(m0.state_dict())
    ctrl = multilora.attach(m1, n=n, r=2, alpha=4)
    return m0, m1, ctrl


def test_fresh_adapters_equal_the_base_model():
    m0, m1, ctrl = _pair()
    x = torch.randn(3, 5, 8)
    ctrl.set_rows([0, 1, 2])
    assert torch.allclose(m0(x), m1(x), atol=1e-6)                   # B = 0
    ctrl.set_rows(None)
    assert torch.equal(m0(x), m1(x))
    assert not isinstance(m1.other, multilora.MultiLoRALinear)         # only target names are wrapped
    assert all(not p.requires_grad for n, p in m1.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))


def test_per_row_index_equals_running_each_adapter_alone():
    _, m, ctrl = _pair()
    for mod in m.modules():
        if isinstance(mod, multilora.MultiLoRALinear):
            nn.init.normal_(mod.B, std=0.1)
    x = torch.randn(4, 5, 8)
    ctrl.set_rows([2, 0, -1, 1])
    y = m(x)
    for r, a in enumerate([2, 0, -1, 1]):
        ctrl.set_rows([a])
        assert torch.allclose(y[r:r + 1], m(x[r:r + 1]), atol=1e-5)
    ctrl.set_rows(None)
    assert torch.allclose(y[2:3], m(x[2:3]), atol=1e-5)                # -1 = base for that row


def test_gradients_reach_only_the_selected_adapter():
    _, m, ctrl = _pair()
    for mod in m.modules():
        if isinstance(mod, multilora.MultiLoRALinear):
            nn.init.normal_(mod.B, std=0.1)
    ctrl.set_rows([1, 1])
    m(torch.randn(2, 3, 8)).sum().backward()
    g = m.q_proj.A.grad
    assert g[1].abs().sum() > 0 and g[0].abs().sum() == 0 and g[2].abs().sum() == 0


def test_save_load_round_trip(tmp_path):
    _, m, ctrl = _pair()
    for mod in m.modules():
        if isinstance(mod, multilora.MultiLoRALinear):
            nn.init.normal_(mod.B, std=0.1)
    multilora.save(tmp_path / "a.pt", m, {"key": "k"})
    fresh = Tiny()
    fresh.load_state_dict({k.replace(".base", ""): v for k, v in m.state_dict().items() if not (k.endswith(".A") or k.endswith(".B"))})
    ctrl2, meta = multilora.load(tmp_path / "a.pt", fresh)
    x = torch.randn(2, 3, 8)
    ctrl.set_rows([0, 2]); ctrl2.set_rows([0, 2])
    assert meta == {"key": "k"} and torch.allclose(m(x), fresh(x), atol=1e-6)


# ---- bin head (amendment 12) ----------------------------------------------------------------- #
import numpy as np

from wmworld import binhead, semwm


def test_region_label_agrees_with_detector_bins_and_margin():
    pl = semwm.Hyperplanes.for_key(semwm.Key.shipped(), 16)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.normal(scale=0.05, size=3)
        lab = binhead.region_label(p, 0.02)
        if np.min(np.abs(p)) < 0.02:
            assert lab == binhead.NEAR
        else:
            assert lab == pl.bin_of(p)


def test_mask_logprob_is_the_summed_region_probability():
    torch.manual_seed(0)
    logits = torch.randn(5, binhead.N_REGIONS)
    got = binhead.mask_logprob(logits, (2, 6)).exp()
    want = torch.softmax(logits, -1)[:, [2, 6]].sum(-1)
    assert torch.allclose(got, want, atol=1e-6)
    h = binhead.BinHead(8, 16)
    assert h(torch.randn(4, 8), torch.randn(1, 8)).shape == (4, binhead.N_REGIONS)   # one state vs many next tokens broadcasts


def test_gaussian_region_probabilities_sum_to_one_and_follow_the_mean():
    mu = torch.tensor([[0.05, -0.05, 0.05], [0.0, 0.0, 0.0]])
    sd = torch.full_like(mu, 0.01)
    lp = binhead.region_logprobs_from_gaussian(mu, sd, 0.02)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-4)
    assert int(lp[0].argmax()) == 0b101                              # signs (+,-,+) -> bin 5, far from the planes
    assert int(lp[1].argmax()) == binhead.NEAR                       # at the planes -> near


# ---- push head (amendment 15) ---------------------------------------------------------------- #
from wmworld import pushhead


def test_push_standardise_matches_the_push_rule():
    torch.manual_seed(0)
    g = torch.randn(3, 6); w = torch.rand(3, 6); v = torch.ones(3, 6); v[2, 4:] = 0
    z = pushhead.standardise(g, w, v)
    for r in range(3):
        k = int(v[r].sum()); ww = w[r, :k] / w[r, :k].sum()
        mu = (ww * g[r, :k]).sum(); sd = torch.sqrt((ww * (g[r, :k] - mu) ** 2).sum())
        assert torch.allclose(z[r, :k], ((g[r, :k] - mu) / sd).clamp(-1, 1), atol=1e-5)
        assert torch.all(z[r, k:] == 0)
    assert torch.all(pushhead.standardise(torch.ones(1, 4), torch.rand(1, 4), torch.ones(1, 4)) == 0)   # flat gains -> no push


def test_push_head_shapes_and_state_features():
    h = pushhead.PushHead(8)
    assert h(torch.randn(2, 8), torch.randn(2, 5, 8), torch.tensor([0, 7]), torch.randn(2, pushhead.N_STATE)).shape == (2, 5)
    pl = semwm.Hyperplanes.for_key(semwm.Key.shipped(), 16)
    f = pushhead.state_features(np.array([0.03, -0.01, 0.02]), (1, 6), pl, 10)
    assert f.shape == (pushhead.N_STATE,) and abs(f[7] - 10 / 64) < 1e-6
    assert np.all(pushhead.state_features(None, (1, 6), pl, 0) == 0)


# ---- sync-free fast path (2026-09-25) ---------------------------------------------------------- #
def test_single_adapter_fast_path_equals_the_gathered_path():
    _, m, ctrl = _pair(n=4)
    for mod in m.modules():
        if isinstance(mod, multilora.MultiLoRALinear):
            nn.init.normal_(mod.B, std=0.1)
    x = torch.randn(3, 5, 8)
    ctrl.set_rows([2, 2, 2])
    assert ctrl.single == 2
    fast = m(x)
    ctrl.set_rows([2, 2, 2]); ctrl.single = None                    # force the general (gather) path
    slow = m(x)
    assert torch.allclose(fast, slow, atol=1e-5)
    ctrl.set_rows([2, -1, 1])                                        # mixed rows still per-row, -1 = base
    assert ctrl.single is None and ctrl.any_on and not ctrl.all_on
    y = m(x)
    for r, a in enumerate([2, -1, 1]):
        ctrl.set_rows([a]); assert torch.allclose(y[r:r + 1], m(x[r:r + 1]), atol=1e-5)
    ctrl.set_rows([-1, -1, -1])
    m0 = Tiny(); m0.load_state_dict({k.replace(".base", ""): v for k, v in m.state_dict().items() if not (k.endswith(".A") or k.endswith(".B"))})
    assert torch.equal(m(x), m0(x))                                  # all rows off = base exactly
